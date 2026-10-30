// B10 kernel (BRISC): out[i] = in[i] * mul + add for n uint32 words held in one DRAM page.
// The same source runs on both chips; each chip's program passes its own runtime args.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2);
    const uint32_t mul = get_arg_val<uint32_t>(3);
    const uint32_t add = get_arg_val<uint32_t>(4);
    constexpr auto in_args = TensorAccessorArgs<0>();
    const auto in = TensorAccessor(in_args, in_addr);
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, out_addr);

    constexpr uint32_t cb = tt::CBIndex::c_0;
    cb_reserve_back(cb, 1);
    const uint32_t l1 = get_write_ptr(cb);
    noc_async_read_page(0, in, l1);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t i = 0; i < n; i++) {
        p[i] = p[i] * mul + add;
    }
    (void)p[n - 1];  // read back the last store before the NoC reads the page (see add_2_integers_in_riscv)
    noc_async_write_page(0, out, l1);
    noc_async_write_barrier();
}
