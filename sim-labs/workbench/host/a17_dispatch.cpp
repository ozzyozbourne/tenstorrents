// A17 lab: launch the add_2_integers kernel N times, optionally via a captured trace or with 2 CQs + events.
//   A17_ITERS=N   number of launches (default 1)
//   A17_TRACE=1   capture the workload once in a trace, then replay it N times (fast dispatch only)
//   A17_CQS=2     open 2 command queues: IO on CQ1, the workload on CQ0, synchronised with events
//   A17_ETH=1     put the dispatch kernels on Ethernet cores instead of the default Tensix (WORKER) cores
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;

static int env_int(const char* name, int dflt) {
    const char* v = std::getenv(name);
    return v ? std::atoi(v) : dflt;
}

int main() {
    const int iters = env_int("A17_ITERS", 1);
    const bool use_trace = env_int("A17_TRACE", 0) != 0;
    const int num_cqs = env_int("A17_CQS", 1);
    const bool eth_dispatch = env_int("A17_ETH", 0) != 0;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        0, DEFAULT_L1_SMALL_SIZE, /*trace_region_size=*/use_trace ? (1 << 20) : DEFAULT_TRACE_REGION_SIZE, num_cqs,
        eth_dispatch ? DispatchCoreConfig{DispatchCoreType::ETH} : DispatchCoreConfig{});
    auto& work_cq = mesh_device->mesh_command_queue(0);
    auto& io_cq = mesh_device->mesh_command_queue(num_cqs - 1);
    auto grid = mesh_device->compute_with_storage_grid_size();
    std::cout << "A17 compute grid " << grid.x << "x" << grid.y << ", hw CQs " << int(mesh_device->num_hw_cqs())
              << std::endl;

    distributed::DeviceLocalBufferConfig dram{.page_size = 4, .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig l1{.page_size = 4, .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig cfg{.size = 4};
    auto s0 = distributed::MeshBuffer::create(cfg, dram, mesh_device.get());
    auto s1 = distributed::MeshBuffer::create(cfg, dram, mesh_device.get());
    auto d = distributed::MeshBuffer::create(cfg, dram, mesh_device.get());
    auto s0l = distributed::MeshBuffer::create(cfg, l1, mesh_device.get());
    auto s1l = distributed::MeshBuffer::create(cfg, l1, mesh_device.get());
    auto dl = distributed::MeshBuffer::create(cfg, l1, mesh_device.get());

    Program program = CreateProgram();
    CoreCoord core = {0, 0};
    KernelHandle k = CreateKernel(
        program,
        "tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(
        program,
        k,
        core,
        {uint32_t(s0->address()),
         uint32_t(s1->address()),
         uint32_t(d->address()),
         uint32_t(s0l->address()),
         uint32_t(s1l->address()),
         uint32_t(dl->address())});
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));

    std::optional<distributed::MeshTraceId> tid;
    if (use_trace) {
        // Warm-up run compiles and loads the binaries; a trace may not do that inside the capture.
        distributed::EnqueueMeshWorkload(work_cq, workload, true);
        tid = mesh_device->begin_mesh_trace(work_cq);
        distributed::EnqueueMeshWorkload(work_cq, workload, false);
        mesh_device->end_mesh_trace(work_cq, *tid);
        std::cout << "A17 trace captured" << std::endl;
    }

    int bad = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
        std::vector<uint32_t> a = {uint32_t(14 + i)}, b = {uint32_t(7 * i + 7)};
        distributed::EnqueueWriteMeshBuffer(io_cq, s0, a, false);
        distributed::EnqueueWriteMeshBuffer(io_cq, s1, b, false);
        if (num_cqs > 1) {
            auto written = io_cq.enqueue_record_event();
            work_cq.enqueue_wait_for_event(written);
        }
        if (use_trace) {
            mesh_device->replay_mesh_trace(work_cq, *tid, false);
        } else {
            distributed::EnqueueMeshWorkload(work_cq, workload, false);
        }
        if (num_cqs > 1) {
            auto ran = work_cq.enqueue_record_event();
            io_cq.enqueue_wait_for_event(ran);
        }
        std::vector<uint32_t> r;
        distributed::EnqueueReadMeshBuffer(io_cq, r, d, true);
        if (r[0] != a[0] + b[0]) {
            bad++;
            std::cout << "A17 iter " << i << ": expected " << a[0] + b[0] << " got " << r[0] << std::endl;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "A17 iters=" << iters << " trace=" << use_trace << " cqs=" << num_cqs << " bad=" << bad
              << " loop_ms=" << ms << " ms_per_iter=" << ms / iters << std::endl;
    if (tid) {
        mesh_device->release_mesh_trace(*tid);
    }
    mesh_device->close();
    return bad ? 1 : 0;
}
