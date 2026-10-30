// SPDX-License-Identifier: Apache-2.0
// Lab B10: one MeshWorkload, two programs. The same kernel (b10/add_const.cpp, found through
// TT_METAL_KERNEL_PATH) runs on mesh coordinate (0,0) and (1,0), each with its own runtime args,
// on a MeshBuffer sharded so that each chip holds different input words.
#include <cstdint>
#include <fmt/core.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <vector>

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

int main() {
    auto mesh = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 1)));
    auto& cq = mesh->mesh_command_queue();
    constexpr uint32_t n = 1024, page = n * sizeof(uint32_t);  // one 4 KiB page per chip

    // Global buffer: 2 rows of 1024 words; shard = one row; row r lands on mesh row r.
    DeviceLocalBufferConfig dram{.page_size = page, .buffer_type = BufferType::DRAM};
    distributed::ShardedBufferConfig cfg{.global_size = 2 * page, .global_buffer_shape = {2, n}, .shard_shape = {1, n}};
    auto in = MeshBuffer::create(cfg, dram, mesh.get());
    auto out = MeshBuffer::create(cfg, dram, mesh.get());
    std::vector<uint32_t> src(2 * n);
    for (uint32_t i = 0; i < 2 * n; i++) src[i] = i;  // chip 0 gets 0..1023, chip 1 gets 1024..2047
    EnqueueWriteMeshBuffer(cq, in, src, false);

    // Per-chip constants: chip at row r computes x * mul[r] + add[r].
    const uint32_t mul[2] = {2, 3}, add[2] = {1000, 7};
    MeshWorkload workload;
    for (uint32_t r = 0; r < 2; r++) {
        Program program = CreateProgram();
        CoreCoord core{0, 0};
        CreateCircularBuffer(program, core,
            CircularBufferConfig(page, {{tt::CBIndex::c_0, tt::DataFormat::UInt32}}).set_page_size(tt::CBIndex::c_0, page));
        std::vector<uint32_t> ct;
        TensorAccessorArgs(*in->get_reference_buffer()).append_to(ct);
        TensorAccessorArgs(*out->get_reference_buffer()).append_to(ct);
        auto k = CreateKernel(program, "b10/add_const.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct});
        SetRuntimeArgs(program, k, core, {in->address(), out->address(), n, mul[r], add[r]});
        MeshCoordinate c(r, 0);
        workload.add_program(MeshCoordinateRange(c, c), std::move(program));
    }
    fmt::print("workload: {} programs, in@0x{:x} out@0x{:x} (same address on both chips)\n",
        workload.get_programs().size(), in->address(), out->address());
    EnqueueMeshWorkload(cq, workload, false);
    std::vector<uint32_t> got;
    EnqueueReadMeshBuffer(cq, got, out, true);

    uint32_t bad = 0;
    for (uint32_t i = 0; i < 2 * n; i++) {
        uint32_t r = i / n, want = src[i] * mul[r] + add[r];
        bad += got[i] != want;
    }
    for (uint32_t r = 0; r < 2; r++) {
        fmt::print("chip row {}: x*{}+{}  in[0..2]={} {} {}  out[0..2]={} {} {}  out[1023]={}\n", r, mul[r], add[r],
            src[r * n], src[r * n + 1], src[r * n + 2], got[r * n], got[r * n + 1], got[r * n + 2], got[r * n + n - 1]);
    }
    fmt::print("mismatches: {} of {}\n", bad, 2 * n);
    mesh->close();
    return bad != 0;
}
