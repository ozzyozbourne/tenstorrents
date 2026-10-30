// SPDX-License-Identifier: Apache-2.0
// Lab B10, step 4a: does TT-Fabric come up on the 2-chip simulator?
// Usage: b10_fabric [fabric_config=2]   (2 = FABRIC_1D, 1 = FABRIC_1D_NEIGHBOR_EXCHANGE, 0 = DISABLED)
#include <cstdlib>
#include <fmt/core.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    auto cfg = static_cast<tt::tt_fabric::FabricConfig>(argc > 1 ? std::atoi(argv[1]) : 2);
    tt::tt_fabric::SetFabricConfig(cfg);
    fmt::print("SetFabricConfig({}) done\n", static_cast<int>(cfg));
    auto mesh = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 1)));
    fmt::print("MeshDevice up: {} devices\n", mesh->num_devices());
    fmt::print("routing planes along axis 0: {}\n",
        tt::tt_fabric::experimental::get_number_of_available_routing_planes(*mesh, 0, 0));
    mesh->close();
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    fmt::print("closed\n");
    return 0;
}
