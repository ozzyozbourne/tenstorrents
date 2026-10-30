// Chapter A09 probe: print the allocator's view of DRAM and L1 on the simulated chip.
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_page_mapping.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <map>
#include <fmt/ranges.h>

using namespace tt::tt_metal;

static std::shared_ptr<distributed::MeshBuffer> make(distributed::MeshDevice* d, size_t size, uint32_t page, BufferType t,
                                                    BufferShardingArgs s = {}) {
    distributed::DeviceLocalBufferConfig lc{.page_size = page, .buffer_type = t, .sharding_args = s};
    distributed::ReplicatedBufferConfig rc{.size = size};
    return distributed::MeshBuffer::create(rc, lc, d);
}

int main() {
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    auto* dev = mesh->get_devices()[0];
    const auto& alloc = mesh->allocator();
    fmt::print("DRAM banks={} bank_size={} MiB alignment={} B\n", alloc->get_num_banks(BufferType::DRAM),
               alloc->get_bank_size(BufferType::DRAM) >> 20, alloc->get_alignment(BufferType::DRAM));
    fmt::print("L1 banks={} bank_size={} B ({} KiB) of worker_l1_size={} B, alignment={} B\n",
               alloc->get_num_banks(BufferType::L1), alloc->get_bank_size(BufferType::L1),
               alloc->get_bank_size(BufferType::L1) / 1024, alloc->get_worker_l1_size(),
               alloc->get_alignment(BufferType::L1));
    fmt::print("compute grid = {}x{}\n", mesh->compute_with_storage_grid_size().x, mesh->compute_with_storage_grid_size().y);
    for (uint32_t b = 0; b < 12 && b < (uint32_t)alloc->get_num_banks(BufferType::DRAM); b++) {
        auto lc = dev->logical_core_from_dram_channel(b);
        auto vc = dev->virtual_core_from_logical_core(lc, tt::CoreType::DRAM);
        fmt::print("  DRAM bank {:2} -> logical dram core {} virtual coord {} bank_offset=0x{:x}\n", b, lc.str(), vc.str(),
                   alloc->get_bank_offset(BufferType::DRAM, b));
    }
    fmt::print("worker logical (0,0) (3,0) (0,3) (7,7) -> virtual {} {} {} {}\n", dev->worker_core_from_logical_core({0, 0}).str(),
               dev->worker_core_from_logical_core({3, 0}).str(), dev->worker_core_from_logical_core({0, 3}).str(),
               dev->worker_core_from_logical_core({7, 7}).str());
    for (uint32_t b = 0; b < 16; b++) {
        fmt::print("  L1 bank {} -> logical core {}\n", b, alloc->get_logical_core_from_bank_id(b).str());
    }

    constexpr uint32_t T = 2048;  // one bf16 32x32 tile
    auto d0 = make(mesh.get(), 1 * T, T, BufferType::DRAM);
    auto d1 = make(mesh.get(), 14 * T, T, BufferType::DRAM);
    auto d2 = make(mesh.get(), 1 * T, T, BufferType::DRAM);
    fmt::print("DRAM interleaved: d0 (1 page) @0x{:x}, d1 (14 pages) @0x{:x}, d2 (1 page) @0x{:x}\n", d0->address(),
               d1->address(), d2->address());
    auto l0 = make(mesh.get(), 1 * T, T, BufferType::L1);
    auto l1 = make(mesh.get(), 100 * T, T, BufferType::L1);
    auto l2 = make(mesh.get(), 1 * T, T, BufferType::L1);
    fmt::print("L1 interleaved:   l0 (1 page) @0x{:x}, l1 (100 pages) @0x{:x}, l2 (1 page) @0x{:x}\n", l0->address(),
               l1->address(), l2->address());

    struct Cfg { const char* name; TensorMemoryLayout lay; CoreRange cores; std::array<uint32_t, 2> shard; ShardOrientation o; };
    std::vector<Cfg> cfgs = {
        {"HEIGHT", TensorMemoryLayout::HEIGHT_SHARDED, CoreRange({0, 0}, {3, 0}), {32, 128}, ShardOrientation::ROW_MAJOR},
        {"WIDTH ", TensorMemoryLayout::WIDTH_SHARDED, CoreRange({0, 0}, {3, 0}), {128, 32}, ShardOrientation::ROW_MAJOR},
        {"BLOCK ", TensorMemoryLayout::BLOCK_SHARDED, CoreRange({0, 0}, {1, 1}), {64, 64}, ShardOrientation::ROW_MAJOR},
        {"BLOCK/COL_MAJOR", TensorMemoryLayout::BLOCK_SHARDED, CoreRange({0, 0}, {1, 1}), {64, 64}, ShardOrientation::COL_MAJOR},
    };
    for (auto& c : cfgs) {
        ShardSpecBuffer spec(CoreRangeSet(c.cores), c.shard, c.o, {32, 32}, {4, 4});
        auto mb = make(mesh.get(), 16 * T, T, BufferType::L1, BufferShardingArgs(spec, c.lay));
        auto* buf = mb->get_backing_buffer();
        auto m = buf->get_buffer_page_mapping();
        std::map<std::string, std::vector<uint32_t>> per;
        for (auto p : *m) per[m->all_cores[p.core_id].str()].push_back(p.host_page);
        fmt::print("{} sharded 4x4 tiles, shard {}x{} elems, @0x{:x}:\n", c.name, c.shard[0], c.shard[1], buf->address());
        for (auto& [core, pages] : per) fmt::print("   core {} holds tiles {}\n", core, pages);
    }
    mesh->close();
    return 0;
}
