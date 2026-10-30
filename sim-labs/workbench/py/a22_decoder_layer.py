# Chapter A22: one tiny Llama-style decoder layer (plus embedding, final norm, LM head and argmax)
# built from the same ttnn ops tt_transformers uses, run as a prefill and then one decode step,
# and checked against a float32 PyTorch reference by PCC.
#
#   python a22_decoder_layer.py [--precision perf|bf16]
#
# perf (default): the dtypes and fidelities of tt_transformers' "performance" setting
#   (models/tt_transformers/tt/model_config.py: ModelOptimizations): wqkv/wo/w2/KV cache bfloat8_b,
#   w1/w3 bfloat4_b with LoFi, other matmuls HiFi2, SDPA prefill HiFi4.
# bf16: every weight and the KV cache in bfloat16, every op HiFi4 (the top of the precision ladder).
# One deliberate difference: Dst accumulates in FP16 everywhere (fp32_dest_acc_en=False), because ttsim
# cannot run kernels built with FP32 Dst accumulation (see the comment at HIFI4 below).
#
# Prefill keeps activations DRAM-interleaved; decode keeps them width-sharded in L1 and uses
# DRAM-sharded matmuls, exactly the split tech_reports/LLMs/llms.md describes.
import argparse, math, time
import torch
import ttnn
from models.tt_transformers.tt.rope import RotarySetup

# ttsim workaround: from_torch(float32 tensor, dtype=bfloat16, mesh_mapper=...) converts on the device and
# ttsim stops with "UndefinedBehavior: tensix_unpacr ... in_data_format=0" (Float32). RotarySetup does exactly
# that, so cast float32 -> bfloat16 on the host first. The values are identical either way.
_from_torch = ttnn.from_torch
def _from_torch_host_cast(t, *a, **kw):
    if isinstance(t, torch.Tensor) and t.dtype == torch.float32 and kw.get("dtype") == ttnn.bfloat16:
        t = t.bfloat16()
    return _from_torch(t, *a, **kw)
ttnn.from_torch = _from_torch_host_cast

ap = argparse.ArgumentParser()
ap.add_argument("--precision", choices=["perf", "bf16"], default="perf")
args = ap.parse_args()
PERF = args.precision == "perf"

# ---- a tiny Llama: every width is a multiple of 12 tiles so it divides over the 12 WH DRAM banks
V, DIM, NH, NKV, HD, FFN = 768, 384, 12, 6, 32, 768      # vocab, model dim, heads, kv heads, head dim, MLP hidden
EPS, THETA = 1e-5, 500000.0
BLOCK, NBLOCKS, USERS = 32, 8, 2                          # paged KV cache: 8 blocks of 32 tokens, 2 users
PROMPT = [64, 32]                                         # prompt length of each user (prefill)
torch.manual_seed(0)

def W(*shape):  # weight with ~unit-variance outputs
    return torch.randn(*shape) / math.sqrt(shape[0])
emb = torch.randn(V, DIM)
g_attn, g_ffn, g_out = (1 + 0.1 * torch.randn(DIM) for _ in range(3))
wq, wk, wv, wo = W(DIM, NH * HD), W(DIM, NKV * HD), W(DIM, NKV * HD), W(NH * HD, DIM)
w1, w3, w2, wlm = W(DIM, FFN), W(DIM, FFN), W(FFN, DIM), W(DIM, V)
prompts = [torch.randint(0, V, (s,)) for s in PROMPT]

# ---------------------------------------------------------------- float32 PyTorch reference
def rms(x, g): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g
def rope(x, pos):  # Meta-style RoPE: rotate interleaved pairs (x0,x1), (x2,x3), ...
    f = 1.0 / THETA ** (torch.arange(0, HD, 2).float() / HD)
    a = pos[:, None].float() * f[None]
    c, s = torch.cos(a).repeat_interleave(2, -1), torch.sin(a).repeat_interleave(2, -1)
    x2 = torch.stack([-x[..., 1::2], x[..., 0::2]], -1).flatten(-2)
    return x * c + x2 * s
def ref_layer(x, pos, cache):  # x [T, DIM] at positions pos; cache = (K, V) lists to append to
    h = rms(x, g_attn)
    q = rope((h @ wq).view(-1, NH, HD).transpose(0, 1), pos)
    k = rope((h @ wk).view(-1, NKV, HD).transpose(0, 1), pos)
    v = (h @ wv).view(-1, NKV, HD).transpose(0, 1)
    cache[0].append(k); cache[1].append(v)
    K, Vv = torch.cat(cache[0], 1), torch.cat(cache[1], 1)
    K, Vv = K.repeat_interleave(NH // NKV, 0), Vv.repeat_interleave(NH // NKV, 0)   # GQA
    mask = pos[:, None] >= torch.arange(K.shape[1])[None]
    att = torch.nn.functional.scaled_dot_product_attention(q, K, Vv, attn_mask=mask)
    x = x + att.transpose(0, 1).reshape(-1, NH * HD) @ wo
    h = rms(x, g_ffn)
    return x + (torch.nn.functional.silu(h @ w1) * (h @ w3)) @ w2
ref_cache = [([], []) for _ in range(USERS)]
ref_prefill_out, ref_next = [], []
for u in range(USERS):
    y = ref_layer(emb[prompts[u]], torch.arange(PROMPT[u]), ref_cache[u])
    ref_prefill_out.append(y)
    ref_next.append(int((rms(y[-1], g_out) @ wlm).argmax()))

# ---------------------------------------------------------------- device
dev = ttnn.open_device(device_id=0)
def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()
def ckc(fid, approx=False, fp32=False, l1acc=True):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=approx,
                                            fp32_dest_acc_en=fp32, packer_l1_acc=l1acc)
# ttsim limitation: any kernel built with fp32_dest_acc_en=True stops with "UnsupportedFunctionality:
# tensix_unpacr: cfg_context_id=1" (or an unpack_to_dst UndefinedBehavior). tt_transformers turns FP32 Dst
# accumulation on for its HiFi2/HiFi4 configs (model_config.py:939-956); here it stays off (FP16 Dst).
HIFI4 = ckc(ttnn.MathFidelity.HiFi4)
HIFI2 = ckc(ttnn.MathFidelity.HiFi2, approx=True) if PERF else HIFI4                 # model_config.py:939-944
HIFI2_FP16 = ckc(ttnn.MathFidelity.HiFi2) if PERF else HIFI4                          # model_config.py:945-950
LOFI = ckc(ttnn.MathFidelity.LoFi) if PERF else HIFI4                                 # model_config.py:933-938
BF8 = ttnn.bfloat8_b if PERF else ttnn.bfloat16
BF4 = ttnn.bfloat4_b if PERF else ttnn.bfloat16
log = []
def step(name, t, note=""):
    mc = t.memory_config()
    where = f"{mc.buffer_type.name} {mc.memory_layout.name}".replace("BufferType.", "").replace("TensorMemoryLayout.", "")
    log.append((name, tuple(t.shape), t.dtype.name if hasattr(t.dtype, "name") else str(t.dtype), where, note))
    return t

# DRAM-sharded weights: width-sharded over the 12 DRAM banks (create_dram_sharded_mem_config, model_config.py:3760)
dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dev.dram_grid_size().x - 1, 0))})
def dram_sharded(w, dtype):
    k, n = w.shape
    spec = ttnn.ShardSpec(dram_grid, (k, n // dev.dram_grid_size().x), ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)
    return ttnn.from_torch(w[None, None], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
def gamma(g):  # RMSNorm weight as ROW_MAJOR sticks of 32 (llms.md 2.3.1.1)
    return ttnn.from_torch(g.view(1, 1, DIM // 32, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)

t_emb = ttnn.from_torch(emb[None, None], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
t_ga, t_gf, t_go = gamma(g_attn), gamma(g_ffn), gamma(g_out)
wqkv = torch.cat([wq, wk, wv], 1)                        # fused QKV weight, [DIM, (NH+2*NKV)*HD]
t_wqkv, t_wo, t_w2, t_wlm = (dram_sharded(w, BF8) for w in (wqkv, wo, w2, wlm))
t_w1, t_w3 = dram_sharded(w1, BF4), dram_sharded(w3, BF4)

# paged KV cache: [max_num_blocks, n_kv_heads, block_size, head_dim] in DRAM (attention.py:545-600)
perm = torch.randperm(NBLOCKS)
page_table = torch.argsort(perm).reshape(USERS, NBLOCKS // USERS).int()   # same recipe as simple_text_demo.py:163-174
t_pt = ttnn.from_torch(page_table, dtype=ttnn.int32, device=dev)
kv_dtype = BF8
kcache, vcache = (ttnn.from_torch(torch.zeros(NBLOCKS, NKV, BLOCK, HD), dtype=kv_dtype, layout=ttnn.TILE_LAYOUT,
                                  device=dev) for _ in range(2))
rope_setup = RotarySetup(dev, USERS, HD, 128, THETA, None)   # the real tt_transformers RoPE helper
sdpa_cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=32,
                                  exp_approx_mode=False)

# ================================================================ PREFILL (one user at a time, DRAM interleaved)
def mm2d_cfg(S, n):  # matmul 2D (llms.md 4.3.2.1): M split over grid rows, N over grid columns
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8), in0_block_w=1, out_subblock_h=1, out_subblock_w=1,
        per_core_M=math.ceil(S / 32 / 8), per_core_N=math.ceil(n / 32 / 8), transpose_mcast=False,
        fused_activation=None, fuse_batch=True)
t0 = time.time()
prefill_pcc, dev_next, margins = [], [], []
for u in range(USERS):
    S = PROMPT[u]
    ids = ttnn.from_torch(prompts[u][None].int(), dtype=ttnn.uint32, device=dev)
    x = ttnn.reshape(ttnn.embedding(ids, t_emb, layout=ttnn.TILE_LAYOUT), (1, 1, S, DIM))
    rec = u == 0
    if rec: step("embedding", x, "ttnn.embedding: token ids -> rows of the table")
    h = ttnn.rms_norm(x, epsilon=EPS, weight=t_ga, compute_kernel_config=HIFI2_FP16)
    if rec: step("attention_norm", h, "ttnn.rms_norm")
    qkv = ttnn.linear(h, t_wqkv, dtype=ttnn.bfloat16, compute_kernel_config=HIFI2, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                      program_config=mm2d_cfg(S, wqkv.shape[1]))
    if rec: step("wqkv", qkv, f"ttnn.linear, weight {t_wqkv.dtype.name}")
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=NH, num_kv_heads=NKV, transpose_k_heads=False,
                                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if rec: step("create_qkv_heads (Q)", q, "split heads")
    cos = ttnn.slice(rope_setup.cos_matrix_prefill, (0, 0, 0, 0), (1, 1, S, HD))
    sin = ttnn.slice(rope_setup.sin_matrix_prefill, (0, 0, 0, 0), (1, 1, S, HD))
    tm = rope_setup.get_both_trans_mats()["prefill"]
    q = ttnn.experimental.rotary_embedding_llama(q, cos, sin, tm, is_decode_mode=False)
    k = ttnn.experimental.rotary_embedding_llama(k, cos, sin, tm, is_decode_mode=False)
    if rec: step("rotary_embedding_llama (Q)", q, "RoPE as a matmul with a 32x32 transformation tile")
    k8, v8 = ttnn.typecast(k, kv_dtype), ttnn.typecast(v, kv_dtype)
    ttnn.experimental.paged_fill_cache(kcache, k8, t_pt, batch_idx=u)
    ttnn.experimental.paged_fill_cache(vcache, v8, t_pt, batch_idx=u)
    if rec: step("paged_fill_cache (K)", k8, f"written into blocks {page_table[u, :S // BLOCK].tolist()}")
    att = ttnn.transformer.scaled_dot_product_attention(ttnn.typecast(q, BF8), k8, v8, is_causal=True,
                                                        scale=HD ** -0.5, program_config=sdpa_cfg,
                                                        compute_kernel_config=HIFI4)
    if rec: step("scaled_dot_product_attention", att, "flash attention, causal, GQA 12 q / 6 kv heads")
    att = ttnn.experimental.nlp_concat_heads(att, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    o = ttnn.linear(att, t_wo, dtype=ttnn.bfloat16, compute_kernel_config=HIFI2, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=mm2d_cfg(S, DIM))
    x = ttnn.add(x, o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if rec: step("wo + residual add", x, "")
    h = ttnn.rms_norm(x, epsilon=EPS, weight=t_gf, compute_kernel_config=HIFI2_FP16)
    a = ttnn.linear(h, t_w1, dtype=ttnn.bfloat16, compute_kernel_config=LOFI, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=mm2d_cfg(S, FFN))
    b = ttnn.linear(h, t_w3, dtype=ttnn.bfloat16, compute_kernel_config=LOFI, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=mm2d_cfg(S, FFN))
    if rec: step("w1 / w3", a, f"weight {t_w1.dtype.name}")
    m = ttnn.mul(a, b, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=BF8,
                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if rec: step("silu(w1) * w3", m, "one fused eltwise op")
    y = ttnn.linear(m, t_w2, dtype=ttnn.bfloat16, compute_kernel_config=HIFI2_FP16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=mm2d_cfg(S, DIM))
    x = ttnn.add(x, y, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if rec: step("w2 + residual add", x, "layer output")
    prefill_pcc.append(pcc(ttnn.to_torch(x)[0, 0], ref_prefill_out[u]))
    last = ttnn.slice(x, (0, 0, S - 32, 0), (1, 1, S, DIM))          # only the last tile goes to the LM head
    logits = ttnn.linear(ttnn.rms_norm(last, epsilon=EPS, weight=t_go), t_wlm, dtype=ttnn.bfloat16,
                         compute_kernel_config=HIFI2, memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=mm2d_cfg(32, V))
    tok = ttnn.argmax(ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT), dim=-1)
    dev_next.append(int(ttnn.to_torch(tok).flatten()[-1]))
    ref_top = (rms(ref_prefill_out[u][-1], g_out) @ wlm).topk(2)
    margins.append((ref_top.indices.tolist(), [round(v, 3) for v in ref_top.values.tolist()]))
    if rec: step("final norm + LM head (last tile)", logits, "then ttnn.argmax on device")
t_prefill = time.time() - t0

print(f"== a22 tiny decoder layer, precision={args.precision}, arch={ttnn.get_arch_name()} ==")
print(f"config: vocab={V} dim={DIM} heads={NH} kv_heads={NKV} head_dim={HD} ffn={FFN}; paged KV {NBLOCKS}x{BLOCK} tokens")
print(f"page_table (virtual block -> physical block, one row per user): {page_table.tolist()}")
print("\n-- prefill (user 0, S=64): op, output shape, dtype, where it lives --")
for n, s, d, w, note in log:
    print(f"  {n:34s} {str(s):18s} {d:11s} {w:24s} {note}")
for u in range(USERS):
    print(f"prefill user {u}: S={PROMPT[u]:3d}  layer-output PCC={prefill_pcc[u]:.5f}  "
          f"next token device={dev_next[u]} torch={ref_next[u]} (torch top-2 {margins[u][0]} logits {margins[u][1]})")
# Teacher forcing: the reference decodes the token the device chose, so the decode PCC compares the same input.
ref_dec = torch.stack([ref_layer(emb[dev_next[u]][None], torch.tensor([PROMPT[u]]), ref_cache[u])[0]
                       for u in range(USERS)])
ref_dec_tok = (rms(ref_dec, g_out) @ wlm).argmax(-1).tolist()
log.clear()

# ================================================================ DECODE (both users at once, L1 width-sharded)
t0 = time.time()
grid = ttnn.CoreGrid(y=2, x=6)                 # 12 tiles of DIM over 12 cores, one 32x32 tile each
act_mc = ttnn.create_sharded_memory_config((32, DIM // grid.num_cores), grid, ttnn.ShardStrategy.WIDTH,
                                           ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
ffn_mc = ttnn.create_sharded_memory_config((32, FFN // grid.num_cores), grid, ttnn.ShardStrategy.WIDTH,
                                           ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
def dram_mm_cfg(k, n):  # dram_matmul_config (model_config.py:3982) for M = one tile of 32 users
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=k // 32 // grid.num_cores, per_core_M=1, per_core_N=math.ceil(n / 32 / grid.num_cores),
        fused_activation=None)
norm_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=[grid.x, grid.y], subblock_w=1,
                                                       block_h=1, block_w=DIM // 32 // grid.num_cores, inplace=False)
pos = torch.tensor(PROMPT + [0] * (32 - USERS), dtype=torch.int32)        # current position of each user slot
t_pos = ttnn.from_torch(pos[:USERS], dtype=ttnn.int32, device=dev)
ids = ttnn.from_torch(torch.tensor([dev_next + [0] * (32 - USERS)]).int(), dtype=ttnn.uint32, device=dev)
x = ttnn.reshape(ttnn.embedding(ids, t_emb, layout=ttnn.TILE_LAYOUT), (1, 1, 32, DIM))
x = step("embedding -> L1 width-sharded", ttnn.to_memory_config(x, act_mc), "32 user rows, 1 tile per core")
h = step("attention_norm (sharded)", ttnn.rms_norm(x, epsilon=EPS, weight=t_ga, program_config=norm_cfg,
                                                   memory_config=act_mc, compute_kernel_config=HIFI2_FP16), "")
qkv = ttnn.linear(h, t_wqkv, program_config=dram_mm_cfg(DIM, wqkv.shape[1]), dtype=ttnn.bfloat16,
                  memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, compute_kernel_config=HIFI2)
step("wqkv (DRAM-sharded matmul)", qkv, "weight streamed from 12 DRAM banks")
qkv = ttnn.sharded_to_interleaved(qkv, ttnn.L1_MEMORY_CONFIG)
# logical batch = real users, padded batch = one tile (attention.py:935-939)
qkv = ttnn.reshape(qkv, (1, 1, USERS, qkv.shape[3]), (1, 1, 32, qkv.shape[3]))
q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(qkv, num_heads=NH, num_kv_heads=NKV,
                                                        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
step("create_qkv_heads_decode (Q)", q, "one user per core")
cos, sin = rope_setup.get_rot_mats(pos[:USERS])
tm = rope_setup.get_both_trans_mats()["decode"]
q = ttnn.experimental.rotary_embedding_llama(q, cos, sin, tm, is_decode_mode=True)
k = ttnn.experimental.rotary_embedding_llama(k, cos, sin, tm, is_decode_mode=True)
step("rotary_embedding_llama decode (Q)", q, "cos/sin fetched by ttnn.embedding(position)")
ttnn.experimental.paged_update_cache(kcache, k, update_idxs_tensor=t_pos, page_table=t_pt)
ttnn.experimental.paged_update_cache(vcache, v, update_idxs_tensor=t_pos, page_table=t_pt)
step("paged_update_cache (K)", k, "one new row per user, block from page_table")
att = ttnn.transformer.paged_scaled_dot_product_attention_decode(
    q, kcache, vcache, page_table_tensor=t_pt, cur_pos_tensor=t_pos, scale=HD ** -0.5,
    program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), exp_approx_mode=False,
                                          q_chunk_size=0, k_chunk_size=0),
    compute_kernel_config=HIFI2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
step("paged_sdpa_decode", att, "flash decode over cached K/V")
att = ttnn.to_memory_config(att, ttnn.create_sharded_memory_config((32, HD), ttnn.CoreRangeSet(
    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(USERS - 1, 0))}), ttnn.ShardStrategy.HEIGHT,
    ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True))
att = ttnn.experimental.nlp_concat_heads_decode(att, num_heads=NH)
att = ttnn.to_memory_config(att, act_mc)
o = ttnn.linear(att, t_wo, program_config=dram_mm_cfg(NH * HD, DIM), dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, compute_kernel_config=HIFI2)
x = step("wo + residual add", ttnn.add(x, o, memory_config=act_mc), "")
h = ttnn.rms_norm(x, epsilon=EPS, weight=t_gf, program_config=norm_cfg, memory_config=act_mc,
                  compute_kernel_config=HIFI2_FP16)
a = ttnn.linear(h, t_w1, program_config=dram_mm_cfg(DIM, FFN), dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, compute_kernel_config=LOFI)
b = ttnn.linear(h, t_w3, program_config=dram_mm_cfg(DIM, FFN), dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, compute_kernel_config=LOFI)
step("w1 / w3 (DRAM-sharded)", a, "")
m = ttnn.mul(a, b, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=BF8, memory_config=a.memory_config())
m = step("silu(w1) * w3", ttnn.to_memory_config(m, ffn_mc), "")
y = ttnn.linear(m, t_w2, program_config=dram_mm_cfg(FFN, DIM), dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, compute_kernel_config=HIFI2_FP16)
x = step("w2 + residual add", ttnn.add(x, y, memory_config=act_mc), "layer output")
logits = ttnn.linear(ttnn.rms_norm(x, epsilon=EPS, weight=t_go, program_config=norm_cfg, memory_config=act_mc),
                     t_wlm, program_config=dram_mm_cfg(DIM, V), dtype=ttnn.bfloat16,
                     memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, compute_kernel_config=HIFI2)
step("final norm + LM head", logits, "")
tok = ttnn.argmax(ttnn.to_layout(ttnn.sharded_to_interleaved(logits, ttnn.L1_MEMORY_CONFIG), ttnn.ROW_MAJOR_LAYOUT),
                  dim=-1)
dec_tok = ttnn.to_torch(tok).flatten()[:USERS].tolist()
dec_pcc = pcc(ttnn.to_torch(x)[0, 0, :USERS], ref_dec)
t_decode = time.time() - t0

print("\n-- decode step (2 users padded to 32 rows): op, output shape, dtype, where it lives --")
for n, s, d, w, note in log:
    print(f"  {n:34s} {str(s):18s} {d:11s} {w:24s} {note}")
print(f"decode (teacher-forced on device tokens {dev_next}): positions={PROMPT}  layer-output PCC={dec_pcc:.5f}  next tokens device={dec_tok} torch={ref_dec_tok}")
kc = ttnn.to_torch(kcache)
for u in range(USERS):
    vb, off = PROMPT[u] // BLOCK, PROMPT[u] % BLOCK
    pb = int(page_table[u, vb])
    nz = [int(b) for b in range(NBLOCKS) if kc[b].abs().sum() > 0]
    print(f"user {u}: token at position {PROMPT[u]} -> virtual block {vb}, row {off} -> physical block {pb}; "
          f"row non-zero: {bool(kc[pb, :, off].abs().sum() > 0)}")
print(f"physical blocks holding data: {sorted(nz)} of {NBLOCKS}")
print(f"wall time: prefill {t_prefill:.1f} s, decode {t_decode:.1f} s (simulator time, not hardware performance)")
ttnn.close_device(dev)
