// B06 writer (NCRISC, NoC 1): drain CB c_16 into the output buffer in DRAM.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_out, 1);                          // a finished tile is visible
        noc_async_write_page(i, out, get_read_ptr(cb_out)); // start the copy to page i in DRAM
        noc_async_write_barrier();                         // it has landed (acked)
        cb_pop_front(cb_out, 1);                           // the page may be reused: acked += 1
    }
}
