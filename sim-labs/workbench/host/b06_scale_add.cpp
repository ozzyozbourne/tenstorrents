// B06 lab: out = relu(a + b) over N bf16 tiles, with our own reader/compute/writer kernels.
// Usage: b06_scale_add [n_tiles=8] [cb_pages=2]
// The kernels are found as "b06/<name>.cpp" through TT_METAL_KERNEL_PATH (or the cwd), see lab B06.
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

using namespace tt::tt_metal;

int main(int argc, char** argv) {
    const uint32_t n_tiles = argc > 1 ? std::atoi(argv[1]) : 8;   // how many 32x32 tiles
    const uint32_t cb_pages = argc > 2 ? std::atoi(argv[2]) : 2;  // pages (tiles) per circular buffer

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);  // one device, here ttsim
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};  // one Tensix core

    // Three DRAM buffers of n_tiles pages; one page = one bf16 tile = 2048 bytes, interleaved over the banks.
    constexpr uint32_t tile_elems = tt::constants::TILE_HW;  // 1024
    constexpr uint32_t tile_bytes = sizeof(bfloat16) * tile_elems;
    distributed::DeviceLocalBufferConfig dram{.page_size = tile_bytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig whole{.size = n_tiles * tile_bytes};
    auto a_buf = distributed::MeshBuffer::create(whole, dram, mesh_device.get());
    auto b_buf = distributed::MeshBuffer::create(whole, dram, mesh_device.get());
    auto out_buf = distributed::MeshBuffer::create(whole, dram, mesh_device.get());

    // Inputs in [-1, 1) from a fixed seed, so every run sees the same numbers and about half the sums are negative.
    std::mt19937 rng(6);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<bfloat16> a(n_tiles * tile_elems), b(n_tiles * tile_elems);
    for (auto& v : a) v = bfloat16(dist(rng));
    for (auto& v : b) v = bfloat16(dist(rng));
    distributed::EnqueueWriteMeshBuffer(cq, a_buf, a, false);
    distributed::EnqueueWriteMeshBuffer(cq, b_buf, b, false);

    // Three circular buffers in the core's L1: inputs c_0 and c_1, output c_16, each cb_pages tiles deep.
    for (tt::CBIndex cb : {tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16}) {
        CreateCircularBuffer(program, core,
            CircularBufferConfig(cb_pages * tile_bytes, {{cb, tt::DataFormat::Float16_b}}).set_page_size(cb, tile_bytes));
    }

    // Kernels: the TensorAccessor descriptions go in as compile-time args, addresses and counts as runtime args.
    std::vector<uint32_t> reader_ct, writer_ct;
    TensorAccessorArgs(*a_buf).append_to(reader_ct);
    TensorAccessorArgs(*b_buf).append_to(reader_ct);
    TensorAccessorArgs(*out_buf).append_to(writer_ct);
    auto reader = CreateKernel(program, "b06/reader.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_ct});
    auto writer = CreateKernel(program, "b06/writer.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_ct});
    auto compute = CreateKernel(program, "b06/compute.cpp", core, ComputeConfig{});
    SetRuntimeArgs(program, reader, core, {a_buf->address(), b_buf->address(), n_tiles});
    SetRuntimeArgs(program, writer, core, {out_buf->address(), n_tiles});
    SetRuntimeArgs(program, compute, core, {n_tiles});

    // Run it and read the result back.
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    std::vector<bfloat16> out;
    distributed::EnqueueReadMeshBuffer(cq, out, out_buf, true);

    // Host golden: relu of the fp32 sum, rounded to bf16 like the device's output format.
    uint32_t exact = 0, off = 0, first_bad = UINT32_MAX;
    float max_err = 0.0f;
    for (uint32_t i = 0; i < out.size(); i++) {
        const float want = static_cast<float>(bfloat16(std::max(0.0f, static_cast<float>(a[i]) + static_cast<float>(b[i]))));
        const float got = static_cast<float>(out[i]);
        const float err = std::fabs(want - got);
        max_err = std::max(max_err, err);
        if (err == 0.0f) {
            exact++;
        } else if (err > 1.0f / 128 * std::max(std::fabs(want), 1.0f / 64)) {  // more than about one bf16 ulp
            off++;
            if (first_bad == UINT32_MAX) first_bad = i;
        }
    }
    fmt::print("n_tiles={} cb_pages={} elements={} exact={} off_by_more_than_1ulp={} max_abs_err={}\n",
               n_tiles, cb_pages, out.size(), exact, off, max_err);
    if (off) fmt::print("first bad element {} (tile {})\n", first_bad, first_bad / tile_elems);
    mesh_device->close();
    fmt::print("{}\n", off == 0 ? "PASS" : "FAIL");
    return off == 0 ? 0 : 1;
}
