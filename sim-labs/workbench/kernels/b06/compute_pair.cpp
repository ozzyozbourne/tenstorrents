// B06 compute, pair variant (lab B06, break B and break C2): out = relu(a + b), two tiles per iteration.
// Not loaded by the host as is. To use it, copy it over your overlay's compute.cpp:
//     cp $W/kernels/b06/compute_pair.cpp $LAB_HOME/kernels/b06/compute.cpp
// n_tiles must be even. With WAIT2 = 2 it is correct on CBs of 2 or 4 pages; set WAIT2 = 1 for break B1.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/relu.h"

constexpr uint32_t WAIT2 = 2;   // correct: 2 (waits are cumulative).  BREAK B1: 1

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    add_init(cb_a, cb_b);
    relu_tile_init();

    for (uint32_t i = 0; i < n_tiles; i += 2) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        tile_regs_acquire();
        add_tiles(cb_a, cb_b, 0, 0, 0);   // pages 0 of c_0 and c_1 -> Dst[0]
        cb_wait_front(cb_a, WAIT2);
        cb_wait_front(cb_b, WAIT2);
        add_tiles(cb_a, cb_b, 1, 1, 1);   // pages 1 -> Dst[1]
        relu_tile(0);
        relu_tile(1);
        tile_regs_commit();
        cb_pop_front(cb_a, 2);
        cb_pop_front(cb_b, 2);
        tile_regs_wait();
        cb_reserve_back(cb_out, 2);
        pack_tile(0, cb_out);
        pack_tile(1, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 2);
    }
}
