// SPDX-License-Identifier: Apache-2.0
// Lab B10, step 4b: send one 4 KiB page from chip 0 to chip 1's DRAM through TT-Fabric (FABRIC_1D).
// Only chip 0 runs a kernel (b10/fabric_send.cpp); chip 1's Ethernet router delivers the page.
#include <cstdint>
#include <fmt/core.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <vector>

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
    auto mesh = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 1)));
    auto& cq = mesh->mesh_command_queue();
    constexpr uint32_t n = 1024, page = n * 4, seed = 0xB10B10u;

    // Same-sized DRAM buffer on both chips (lock-step allocation: same address on each), zeroed.
    DeviceLocalBufferConfig dram{.page_size = page, .buffer_type = BufferType::DRAM};
    auto buf = MeshBuffer::create(ReplicatedBufferConfig{.size = page}, dram, mesh.get());
    EnqueueWriteMeshBuffer(cq, buf, std::vector<uint32_t>(n, 0), true);

    MeshCoordinate c0(0, 0), c1(1, 0);
    auto src = mesh->get_fabric_node_id(c0), dst = mesh->get_fabric_node_id(c1);
    auto links = tt::tt_fabric::get_forwarding_link_indices(*mesh, src, dst);
    fmt::print("fabric nodes: chip0={} chip1={}, forwarding links 0->1: {}\n", src, dst, links.size());

    Program program = CreateProgram();
    CoreCoord core{0, 0};
    CreateCircularBuffer(program, core,
        CircularBufferConfig(page, {{tt::CBIndex::c_0, tt::DataFormat::UInt32}}).set_page_size(tt::CBIndex::c_0, page));
    std::vector<uint32_t> ct;
    TensorAccessorArgs(*buf->get_reference_buffer()).append_to(ct);
    ct.push_back(page);
    auto k = CreateKernel(program, "b10/fabric_send.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct});
    std::vector<uint32_t> rt = {static_cast<uint32_t>(buf->address()), seed};
    tt::tt_fabric::append_fabric_connection_rt_args(src, dst, links.at(0), program, core, rt);
    SetRuntimeArgs(program, k, core, rt);
    MeshWorkload wl;
    wl.add_program(MeshCoordinateRange(c0, c0), std::move(program));
    EnqueueMeshWorkload(cq, wl, true);
    fmt::print("sender kernel on chip 0 finished\n");

    for (auto c : {c0, c1}) {
        std::vector<uint32_t> got;
        ReadShard(cq, got, buf, c, true);
        uint32_t ok = 0;
        for (uint32_t i = 0; i < n; i++) ok += got[i] == (seed ^ (i * 0x9E3779B9u));
        fmt::print("chip at mesh ({},0): word0=0x{:08x} word1=0x{:08x}  {} of {} words match the pattern\n", c[0], got[0],
            got[1], ok, n);
    }
    mesh->close();
    return 0;
}
