// gridinfo (chapter A03): open one device and print its grid sizes and where a few logical
// worker cores land in translated ("virtual") NoC coordinates.
//   cp $W/host/gridinfo.cpp $LAB_HOME/ && $W/build_host.sh $LAB_HOME/gridinfo.cpp && $LAB_HOME/gridinfo
#include <cstdio>
#include <memory>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt::tt_metal;

int main() {
    std::shared_ptr<distributed::MeshDevice> dev = distributed::MeshDevice::create_unit_mesh(0);
    CoreCoord g = dev->grid_size(), l = dev->logical_grid_size();
    CoreCoord c = dev->compute_with_storage_grid_size(), d = dev->dram_grid_size();
    printf("grid_size            = %zux%zu\n", g.x, g.y);
    printf("logical_grid_size    = %zux%zu\n", l.x, l.y);
    printf("compute grid         = %zux%zu\n", c.x, c.y);
    printf("dram_grid_size       = %zux%zu\n", d.x, d.y);
    const CoreCoord probes[] = {{0, 0}, {1, 0}, {0, 1}, {l.x - 1, l.y - 1}};
    for (const CoreCoord& p : probes) {
        CoreCoord v = dev->worker_core_from_logical_core(p);
        printf("logical (%zu,%zu) -> virtual/translated (%zu,%zu)\n", p.x, p.y, v.x, v.y);
    }
    dev->close();
    return 0;
}
