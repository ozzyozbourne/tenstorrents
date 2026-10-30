// SPDX-License-Identifier: Apache-2.0
// Lab B10: open a 2x1 MeshDevice on the wh_x2 simulator and print what tt-metal discovered:
// which chip sits at which mesh coordinate, which one has PCIe (MMIO), and every Ethernet core
// with the chip and core at the other end of its link.
#include <algorithm>
#include <fmt/core.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

int main() {
    auto mesh = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 1)));
    setvbuf(stdout, nullptr, _IONBF, 0);
    fmt::print("mesh shape {}x{}, {} devices\n", mesh->num_rows(), mesh->num_cols(), mesh->num_devices());
    for (uint32_t r = 0; r < mesh->num_rows(); r++) {
        for (uint32_t c = 0; c < mesh->num_cols(); c++) {
            IDevice* d = mesh->get_device(r, c);
            auto grid = d->compute_with_storage_grid_size();
            fmt::print("mesh ({},{}) -> chip {}  mmio={}  worker grid {}x{}  eth cores (logical) {}\n", r, c, d->id(),
                d->is_mmio_capable(), grid.x, grid.y, d->ethernet_cores().size());
            auto active_set = d->get_active_ethernet_cores();
            std::vector<CoreCoord> active(active_set.begin(), active_set.end());
            std::sort(active.begin(), active.end());
            // get_active_ethernet_cores() returns LOGICAL coords (device.cpp:223 -> control_plane.cpp:3134), despite device.hpp:206.
            for (const CoreCoord& l : active) {
                std::string peer = "no trained link (TT_FATAL in get_connected_ethernet_core)";
                try {
                    auto [peer_chip, peer_core] = d->get_connected_ethernet_core(l);
                    peer = fmt::format("chip {} logical {}", peer_chip, peer_core.str());
                } catch (const std::exception&) {
                }
                fmt::print("  active eth  logical {} = NoC {}  <->  {}\n", l.str(),
                    d->ethernet_core_from_logical_core(l).str(), peer);
            }
            fmt::print("  inactive eth cores: {}\n", d->get_inactive_ethernet_cores().size());
            for (IDevice* o : mesh->get_devices()) {
                if (o->id() == d->id()) continue;
                auto sockets = d->get_ethernet_sockets(o->id());
                std::string s;
                for (auto& k : sockets) s += k.str() + " ";
                fmt::print("  ethernet sockets to chip {}: {}\n", o->id(), s);
            }
        }
    }
    mesh->close();
    return 0;
}
