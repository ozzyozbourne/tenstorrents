// B06 compute (TRISC0/1/2): out = relu(a + b), one tile at a time.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/relu.h"

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);  // UNPACK/MATH/PACK configure; MATH SEMINITs MATH_PACK
    add_init(cb_a, cb_b);                            // load the ELWADD unpack and math programs
    relu_tile_init();                                // MATH: set up the SFPU relu

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_a, 1);       // UNPACK: tile i of A is visible
        cb_wait_front(cb_b, 1);       // UNPACK: tile i of B is visible

        tile_regs_acquire();          // MATH: SEMWAIT until a Dst half is free
        add_tiles(cb_a, cb_b, 0, 0, 0);  // UNPACK: SrcA <- A, SrcB <- B;  MATH: Dst[0] = SrcA + SrcB (FPU)
        relu_tile(0);                 // MATH: Dst[0] = max(Dst[0], 0) on the SFPU
        tile_regs_commit();           // MATH: SEMPOST, the half belongs to PACK now

        cb_pop_front(cb_a, 1);        // UNPACK: free the input pages (acked += 1 once unpack is done)
        cb_pop_front(cb_b, 1);

        tile_regs_wait();             // PACK: SEMWAIT until MATH has committed
        cb_reserve_back(cb_out, 1);   // PACK: a free page in c_16
        pack_tile(0, cb_out);         // PACK: Dst[0] -> L1 page at c_16's write pointer
        tile_regs_release();          // PACK: ZEROACC the half, SEMGET
        cb_push_back(cb_out, 1);      // PACK: received += 1 once the packer is done
    }
}
