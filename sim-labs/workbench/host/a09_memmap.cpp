// Chapter A09 probe: print the allocator's view of DRAM and L1 on the simulated chip.
//
// What it prints, in order:
//   1. bank counts, bank sizes and alignments for DRAM and L1;
//   2. which NoC tile and address offset each DRAM bank uses (the soc descriptor's dram_views);
//   3. the logical core behind the first 16 L1 banks (the allocator shuffles L1 bank ids);
//   4. the addresses of three DRAM and three L1 interleaved buffers, to show lockstep allocation
//      (DRAM grows bottom-up, L1 grows top-down);
//   5. which core holds which tile of a 4x4-tile L1 buffer under height, width and block sharding.
//
// Build: build_host.sh a09_memmap.cpp   Run: ./a09_memmap   (after sourcing env.sh wh or bh)
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

// Allocate a buffer of `size` bytes cut into `page_size`-byte pages. Interleaved by default;
// pass `sharding` to shard it instead.
static std::shared_ptr<distributed::MeshBuffer> make_buffer(
    distributed::MeshDevice* mesh, size_t size, uint32_t page_size, BufferType type, BufferShardingArgs sharding = {}) {
    distributed::DeviceLocalBufferConfig local{.page_size = page_size, .buffer_type = type, .sharding_args = sharding};
    distributed::ReplicatedBufferConfig global{.size = size};
    return distributed::MeshBuffer::create(global, local, mesh);
}

int main() {
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    auto* dev = mesh->get_devices()[0];
    const auto& alloc = mesh->allocator();

    // 1. Bank geometry as the allocator sees it.
    fmt::print("DRAM banks={} bank_size={} MiB alignment={} B\n", alloc->get_num_banks(BufferType::DRAM),
               alloc->get_bank_size(BufferType::DRAM) >> 20, alloc->get_alignment(BufferType::DRAM));
    fmt::print("L1 banks={} bank_size={} B ({} KiB) of worker_l1_size={} B, alignment={} B\n",
               alloc->get_num_banks(BufferType::L1), alloc->get_bank_size(BufferType::L1),
               alloc->get_bank_size(BufferType::L1) / 1024, alloc->get_worker_l1_size(),
               alloc->get_alignment(BufferType::L1));
    fmt::print("compute grid = {}x{}\n", mesh->compute_with_storage_grid_size().x, mesh->compute_with_storage_grid_size().y);

    // 2. DRAM bank -> NoC tile and offset. On Wormhole, odd banks are the second GDDR6 channel (+1 GiB).
    for (uint32_t b = 0; b < (uint32_t)alloc->get_num_banks(BufferType::DRAM); b++) {
        auto logical = dev->logical_core_from_dram_channel(b);
        auto virt = dev->virtual_core_from_logical_core(logical, tt::CoreType::DRAM);
        fmt::print("  DRAM bank {:2} -> logical dram core {} virtual coord {} bank_offset=0x{:x}\n", b, logical.str(),
                   virt.str(), alloc->get_bank_offset(BufferType::DRAM, b));
    }
    // Worker cores use translated (virtual) coordinates, not physical NoC tiles.
    fmt::print("worker logical (0,0) (3,0) (0,3) (7,7) -> virtual {} {} {} {}\n", dev->worker_core_from_logical_core({0, 0}).str(),
               dev->worker_core_from_logical_core({3, 0}).str(), dev->worker_core_from_logical_core({0, 3}).str(),
               dev->worker_core_from_logical_core({7, 7}).str());

    // 3. L1 bank id -> core. The order comes from a fixed-seed std::shuffle in the allocator.
    for (uint32_t b = 0; b < 16; b++) {
        fmt::print("  L1 bank {} -> logical core {}\n", b, alloc->get_logical_core_from_bank_id(b).str());
    }

    // 4. Lockstep allocation: every bank reserves ceil(pages / num_banks) pages for each buffer.
    constexpr uint32_t TILE_BYTES = 2048;  // one bf16 32x32 tile = one page
    auto d0 = make_buffer(mesh.get(), 1 * TILE_BYTES, TILE_BYTES, BufferType::DRAM);
    auto d1 = make_buffer(mesh.get(), 14 * TILE_BYTES, TILE_BYTES, BufferType::DRAM);
    auto d2 = make_buffer(mesh.get(), 1 * TILE_BYTES, TILE_BYTES, BufferType::DRAM);
    fmt::print("DRAM interleaved: d0 (1 page) @0x{:x}, d1 (14 pages) @0x{:x}, d2 (1 page) @0x{:x}\n", d0->address(),
               d1->address(), d2->address());
    auto l0 = make_buffer(mesh.get(), 1 * TILE_BYTES, TILE_BYTES, BufferType::L1);
    auto l1 = make_buffer(mesh.get(), 100 * TILE_BYTES, TILE_BYTES, BufferType::L1);
    auto l2 = make_buffer(mesh.get(), 1 * TILE_BYTES, TILE_BYTES, BufferType::L1);
    fmt::print("L1 interleaved:   l0 (1 page) @0x{:x}, l1 (100 pages) @0x{:x}, l2 (1 page) @0x{:x}\n", l0->address(),
               l1->address(), l2->address());

    // 5. Sharding: a 128x128 bf16 tensor (4x4 tiles) under each layout. Shard shapes are in elements.
    struct ShardCase {
        const char* name;
        TensorMemoryLayout layout;
        CoreRange cores;
        std::array<uint32_t, 2> shard_shape;  // {height, width}
        ShardOrientation orientation;
    };
    const std::vector<ShardCase> cases = {
        {"HEIGHT", TensorMemoryLayout::HEIGHT_SHARDED, CoreRange({0, 0}, {3, 0}), {32, 128}, ShardOrientation::ROW_MAJOR},
        {"WIDTH ", TensorMemoryLayout::WIDTH_SHARDED, CoreRange({0, 0}, {3, 0}), {128, 32}, ShardOrientation::ROW_MAJOR},
        {"BLOCK ", TensorMemoryLayout::BLOCK_SHARDED, CoreRange({0, 0}, {1, 1}), {64, 64}, ShardOrientation::ROW_MAJOR},
        {"BLOCK/COL_MAJOR", TensorMemoryLayout::BLOCK_SHARDED, CoreRange({0, 0}, {1, 1}), {64, 64}, ShardOrientation::COL_MAJOR},
    };
    for (const auto& c : cases) {
        // Page shape is one 32x32 tile; the tensor is 4x4 pages.
        ShardSpecBuffer spec(CoreRangeSet(c.cores), c.shard_shape, c.orientation, {32, 32}, {4, 4});
        auto mesh_buf = make_buffer(mesh.get(), 16 * TILE_BYTES, TILE_BYTES, BufferType::L1, BufferShardingArgs(spec, c.layout));
        auto* buf = mesh_buf->get_backing_buffer();
        // Group host page ids (tile ids in row-major order) by the core that stores them.
        auto mapping = buf->get_buffer_page_mapping();
        std::map<std::string, std::vector<uint32_t>> tiles_per_core;
        for (auto page : *mapping) tiles_per_core[mapping->all_cores[page.core_id].str()].push_back(page.host_page);
        fmt::print("{} sharded 4x4 tiles, shard {}x{} elems, @0x{:x}:\n", c.name, c.shard_shape[0], c.shard_shape[1],
                   buf->address());
        for (const auto& [core, tiles] : tiles_per_core) fmt::print("   core {} holds tiles {}\n", core, tiles);
        // mesh_buf goes out of scope here, so the next case reuses the same L1 address.
    }
    mesh->close();
    return 0;
}
