# A custom op with no C++: (a - b)^2 on one Tensix core, built from a ProgramDescriptor and run by ttnn.generic_op.
import torch, ttnn

READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);   // runtime args: change every call
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t n_tiles = get_arg_val<uint32_t>(2);
    constexpr auto a_args = TensorAccessorArgs<0>();     // compile-time args: baked into the binary
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto a = TensorAccessor(a_args, a_addr);
    const auto b = TensorAccessor(b_args, b_addr);
    const uint32_t page = get_local_cb_interface(0).fifo_page_size;
    Noc noc;
    DataflowBuffer cb_a(0), cb_b(1);
    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_a.reserve_back(1); cb_b.reserve_back(1);
        noc.async_read(a, cb_a, page, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read(b, cb_b, page, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_a.push_back(1); cb_b.push_back(1);
    }
}
"""

COMPUTE = r"""
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"
void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0, 1, 16);
    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(0, 1); cb_wait_front(1, 1); cb_reserve_back(16, 1);
        tile_regs_acquire();
        sub_init(0, 1);
        sub_tiles(0, 1, 0, 0, 0);   // FPU: Dst[0] = a - b
        square_tile_init();
        square_tile(0);             // SFPU: Dst[0] = Dst[0]^2
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, 16);
        tile_regs_release();
        cb_push_back(16, 1); cb_pop_front(0, 1); cb_pop_front(1, 1);
    }
}
"""

def sqdiff(a, b, out):
    n_tiles = a.logical_volume() // (32 * 32)
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    tile_bytes = 2 * 32 * 32  # bf16
    def cb(idx):
        return ttnn.CBDescriptor(total_size=2 * tile_bytes, core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=tile_bytes)])
    rd_rt, wr_rt, cp_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rd_rt[0][0] = [a.buffer_address(), b.buffer_address(), n_tiles]
    wr_rt[0][0] = [out.buffer_address(), n_tiles, 0]
    cp_rt[0][0] = [n_tiles]
    reader = ttnn.KernelDescriptor(kernel_source=READER, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core,
        compile_time_args=ttnn.TensorAccessorArgs(a).get_compile_time_args() + ttnn.TensorAccessorArgs(b).get_compile_time_args(),
        runtime_args=rd_rt, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=core,
        compile_time_args=[16] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
        runtime_args=wr_rt, config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(kernel_source=COMPUTE, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core, compile_time_args=[], runtime_args=cp_rt, config=ttnn.ComputeConfigDescriptor())
    desc = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=[cb(0), cb(1), cb(16)])
    return ttnn.generic_op([a, b, out], desc)

torch.manual_seed(0)
dev = ttnn.open_device(device_id=0)
def T(t): return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
for call in range(3):
    ta, tb = torch.rand(64, 64, dtype=torch.bfloat16), torch.rand(64, 64, dtype=torch.bfloat16)
    a, b = T(ta), T(tb)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([64, 64]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    res = ttnn.to_torch(sqdiff(a, b, out)).float()
    ref = (ta.float() - tb.float()) ** 2
    pcc = torch.corrcoef(torch.stack([res.flatten(), ref.flatten()]))[0, 1].item()
    print(f"call {call}: a@0x{a.buffer_address():x} out@0x{out.buffer_address():x} "
          f"cache entries={dev.num_program_cache_entries()} max|err|={(res-ref).abs().max().item():.4f} PCC={pcc:.5f}")
ttnn.close_device(dev)
