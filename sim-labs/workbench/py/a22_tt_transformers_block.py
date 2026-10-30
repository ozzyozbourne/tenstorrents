# Chapter A22: try the *real* tt_transformers modules under ttsim, offline, with dummy weights.
# Builds ModelArgs for Llama-3.2-1B-Instruct from the config shipped in models/tt_transformers/model_params
# (no download), keeps 1 decoder layer, builds TransformerBlock, and runs one decode step at batch 1
# against the HuggingFace reference layer (models/tt_transformers/tests/test_decoder.py, trimmed).
# Prints how far it got.
#
#   HF_MODEL=meta-llama/Llama-3.2-1B-Instruct python a22_tt_transformers_block.py              # stock configs
#   A22_FP16_DST=1 HF_MODEL=meta-llama/Llama-3.2-1B-Instruct python a22_tt_transformers_block.py # FP16 Dst
import os, sys, time, torch, ttnn

# Same ttsim workaround as a22_decoder_layer.py: cast float32 -> bf16 on the host before from_torch.
_from_torch = ttnn.from_torch
def _cast(t, *a, **kw):
    if isinstance(t, torch.Tensor) and t.dtype == torch.float32 and kw.get("dtype") == ttnn.bfloat16:
        t = t.bfloat16()
    return _from_torch(t, *a, **kw)
ttnn.from_torch = _cast

# ttsim cannot run kernels built with fp32_dest_acc_en=True (UnsupportedFunctionality: tensix_unpacr
# cfg_context_id=1, or UndefinedBehavior unpack_to_dst ... in_data_format=0). tt_transformers sets it on for its
# HiFi2/HiFi4/SDPA configs (model_config.py:939-986). With A22_FP16_DST=1 every compute config built after this
# point is forced to FP16 Dst accumulation, which changes numerics slightly but lets the block run.
if os.environ.get("A22_FP16_DST"):
    _WCKC = ttnn.WormholeComputeKernelConfig
    def _fp16_dst(*a, **kw):
        kw["fp32_dest_acc_en"] = False
        return _WCKC(*a, **kw)
    ttnn.WormholeComputeKernelConfig = _fp16_dst

from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode, PagedAttentionConfig
from models.tt_transformers.tt.decoder import TransformerBlock
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.rope import RotarySetup

os.environ.setdefault("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
stage = "open device"
t0 = time.time()
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    stage = "ModelArgs(dummy_weights=True)"
    args = ModelArgs(mesh, dummy_weights=True, max_batch_size=1, max_seq_len=256)
    args.n_layers = 1
    print(f"ok  {stage}: {args.model_name} dim={args.dim} heads={args.n_heads} kv_heads={args.n_kv_heads} "
          f"hidden={args.hidden_dim} device_name={args.device_name}", flush=True)
    stage = "load_state_dict (random weights from the HF config)"
    torch.manual_seed(42)        # dummy weights are a random init; seed it so the reference gets the same one
    sd = args.load_state_dict()
    print(f"ok  {stage}: {len(sd)} tensors", flush=True)
    stage = "RotarySetup"
    rope = RotarySetup(mesh, 1, args.head_dim, args.max_seq_len, args.rope_theta, args.rope_scaling, args.use_qk_fused)
    print(f"ok  {stage}", flush=True)
    stage = "TransformerBlock (weights -> DRAM, bfloat8_b)"
    pac = PagedAttentionConfig(block_size=32, max_num_blocks=8)
    pt = torch.argsort(torch.randperm(8)).reshape(1, 8)
    pt_tt = ttnn.from_torch(pt, device=mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    blk = TransformerBlock(args=args, mesh_device=mesh, tt_ccl=TT_CCL(mesh), dtype=ttnn.bfloat8_b, state_dict=sd,
                           layer_num=0, weight_cache_path=None, transformation_mats=rope.get_both_trans_mats(),
                           paged_attention_config=pac)
    print(f"ok  {stage}  ({time.time() - t0:.0f} s so far)", flush=True)
    stage = "reference_decoder"
    torch.manual_seed(42)        # same seed -> the HF reference layer is initialised with the same random weights
    ref = args.reference_decoder(load_checkpoint=False)
    # load_state_dict renames HF keys to Meta ones (and permutes wq/wk for RoPE); wv is copied as is.
    k = next(k for k in sd if k.endswith("attention.wv.weight"))
    same = torch.equal(ref.decoder.self_attn.v_proj.weight.float(), sd[k].float())
    print(f"ok  {stage}: v_proj weights identical to the device copy ({k}): {same}", flush=True)
    stage = "one decode step"
    from models.tt_transformers.tests.test_utils import get_ref_model_dype
    x = torch.rand(1, 1, args.dim, dtype=get_ref_model_dype(ref, args.model_name)) * 2 - 1
    pos = torch.tensor([0])
    pos_tt = ttnn.from_torch(pos, device=mesh, dtype=ttnn.int32)
    xin = args.prepare_residual_tensor_decode(x.clone(), args.get_residual_mem_config(Mode.DECODE, None))
    out = blk(xin, pos_tt, rot_mats_global=rope.get_rot_mats(pos), mode=Mode.DECODE, page_table=pt_tt)
    out = ttnn.to_torch(out)[:, 0:1, :1, : args.dim].view(-1, 1, args.dim)
    print(f"ok  {stage}  ({time.time() - t0:.0f} s so far)", flush=True)
    stage = "reference layer (HF, CPU)"
    from models.tt_transformers.tt.common import precompute_freqs
    cos, sin = precompute_freqs(args.head_dim, args.max_seq_len * 2, args.rope_theta,
                                args.rope_scaling.factor if args.rope_scaling else None,
                                args.rope_scaling.original_max_position_embeddings if args.rope_scaling else None,
                                args.rope_scaling.rope_type.value if args.rope_scaling else "llama3")
    refo = ref(x, pos[0], torch.complex(cos, sin)[pos[0], :].unsqueeze(0), mask=None)
    refo = refo.reshape(-1).float(); o = out.reshape(-1).float()[: refo.numel()]
    p = torch.corrcoef(torch.stack([o, refo]))[0, 1].item()
    print(f"ok  {stage}: decode-step PCC vs HF reference = {p:.5f}")
except Exception as e:
    print(f"FAILED at: {stage}: {type(e).__name__}: {(str(e).splitlines() or [repr(e)])[0][:300]}")
    sys.exit(1)
finally:
    ttnn.close_mesh_device(mesh)
