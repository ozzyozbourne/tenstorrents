// B08 compute kernel (TRISC0/1/2): out = f(in), one tile at a time, f chosen by a define from the host
// (b08_sfpu.cpp). Copy this directory into $LAB_HOME/kernels so the host finds "b08/sfpu_lab.cpp".
//   B08_FN_COPY       nothing: unpack -> Dst -> pack (format experiments)
//   B08_FN_MAP        each element becomes 32*vector + lane: shows which SFPU lane touched which Dst cell
//   B08_FN_PIECEWISE  four-region "soft clamp" with v_if / v_elseif / v_elseif / v_else
//   B08_FN_SIN        degree-7 Taylor sin; with B08_FOLD=1 first folds |x| > pi/2 into [-pi/2, pi/2]
//   B08_FN_EXP        built-in exp_tile<B08_EXPT>
//   B08_FN_RECIP      built-in recip_tile (approximation follows math_approx_mode, i.e. APPROX)
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/compute_kernel_api.h"

#ifdef TRISC_MATH
#include "sfpi.h"
namespace ckernel::sfpu {

// One call = one 16x16 face = 8 vectors of 32 lanes. The LLK calls it 4 times per tile.
inline void b08_map_face() {
    for (int i = 0; i < 8; i++) {
        // LReg15 (vConstTileId) holds 2*lane; turn it into a float and halve it.
        sfpi::vFloat lane = sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(sfpi::vConstTileId), sfpi::RoundMode::Nearest);
        sfpi::dst_reg[i] = lane * 0.5f + float(32 * i);
    }
}

inline void b08_piecewise_face() {
    for (int i = 0; i < 8; i++) {
        sfpi::vFloat x = sfpi::dst_reg[i];
        sfpi::vFloat y;
        v_if (x < -1.0f) {             // region A: flat at -1
            y = -1.0f;
        } v_elseif (x < 0.0f) {        // region B: x(x+2), from -1 up to 0
            y = x * (x + 2.0f);
        } v_elseif (x < 1.0f) {        // region C: x(2-x), from 0 up to 1
            y = x * (2.0f - x);
        } v_else {                     // region D: flat at +1
            y = 1.0f;
        }
        v_endif;
        sfpi::dst_reg[i] = y;
    }
}

inline void b08_sin_face() {
    for (int i = 0; i < 8; i++) {
        sfpi::vFloat x = sfpi::dst_reg[i];
#if B08_FOLD
        v_if (x > 1.57079637f) {           // sin(x) = sin(pi - x)
            x = 3.14159274f - x;
        } v_elseif (x < -1.57079637f) {    // sin(x) = sin(-pi - x)
            x = -3.14159274f - x;
        }
        v_endif;
#endif
        sfpi::vFloat x2 = x * x;           // Horner: x(1 + x2(-1/6 + x2(1/120 + x2(-1/5040))))
        sfpi::vFloat p = x2 * -1.98412698e-4f + 8.33333377e-3f;
        p = p * x2 - 0.166666672f;
        p = p * x2 + 1.0f;
        sfpi::dst_reg[i] = x * p;
    }
}

}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);
#if B08_FN_EXP
    exp_tile_init<B08_EXPT>();
#elif B08_FN_RECIP
    recip_tile_init();
#endif

    for (uint32_t t = 0; t < n_tiles; t++) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);   // UNPACK + MATH: tile -> Dst[0]
#if B08_FN_MAP
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, b08_map_face, 0, VectorMode::RC));
#elif B08_FN_PIECEWISE
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, b08_piecewise_face, 0, VectorMode::RC));
#elif B08_FN_SIN
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, b08_sin_face, 0, VectorMode::RC));
#elif B08_FN_EXP
        exp_tile<B08_EXPT>(0);
#elif B08_FN_RECIP
        recip_tile(0);
#endif
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);     // PACK: Dst[0] -> output CB in the CB's data format
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, 1);
        tile_regs_release();
    }
}
