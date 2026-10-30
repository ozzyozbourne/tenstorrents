// B08 lab: run one SFPU function over evenly spaced inputs on one Tensix core and compare with a
// double-precision reference on the host. The compute kernel is "b08/sfpu_lab.cpp" (copy
// sim-labs/overlays/b08/. into $LAB_HOME/kernels). Everything is chosen with environment variables:
//   B08_FN     copy | map | piecewise | sin | exp | recip      (default piecewise)
//   B08_FMT    bf16 | fp32   data format of the DRAM buffers and both CBs (default bf16)
//   B08_DST32  0 | 1         ComputeConfig.fp32_dest_acc_en (default: 1 if B08_FMT=fp32)
//   B08_U2D    0 | 1         unpack c_0 straight to Dst as FP32 (UnpackToDestFp32), skipping SrcA
//   B08_APPROX 0 | 1         ComputeConfig.math_approx_mode, i.e. the APPROX constant in the kernel
//   B08_EXPT   0 | 1         template argument of exp_tile<>/exp_tile_init<> (B08_FN=exp)
//   B08_FOLD   0 | 1         sin: fold |x| > pi/2 first (default 1)
//   B08_LO, B08_HI, B08_TILES   input range and tile count; B08_DUMP=file writes "x ref got" lines
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace tt::tt_metal;

static std::string env(const char* k, const char* d) { const char* v = std::getenv(k); return v ? v : d; }
static double envf(const char* k, double d) { const char* v = std::getenv(k); return v ? std::atof(v) : d; }

static double reference(const std::string& fn, double x) {
    if (fn == "piecewise") return x < -1 ? -1 : x < 0 ? x * (x + 2) : x < 1 ? x * (2 - x) : 1;
    if (fn == "sin") return std::sin(x);
    if (fn == "exp") return std::exp(x);
    if (fn == "recip") return 1.0 / x;
    return x;  // copy
}

// Signed position of a float on the number line in units of the format's last place.
static int64_t ordinal(float f, bool bf16) {
    uint32_t u = std::bit_cast<uint32_t>(f);
    int64_t m = bf16 ? (u >> 16) & 0x7fff : u & 0x7fffffff;
    return (u >> 31) ? -m : m;
}

template <typename T>
static int run(const std::string& fn, bool fp32, bool dst32, bool u2d, bool approx, int expt, int fold,
               double lo, double hi, uint32_t n_tiles) {
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};
    const tt::DataFormat df = fp32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t n = n_tiles * tt::constants::TILE_HW;
    const uint32_t tile_bytes = sizeof(T) * tt::constants::TILE_HW;

    distributed::DeviceLocalBufferConfig dram{.page_size = tile_bytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig whole{.size = n * sizeof(T)};
    auto in_buf = distributed::MeshBuffer::create(whole, dram, mesh_device.get());
    auto out_buf = distributed::MeshBuffer::create(whole, dram, mesh_device.get());

    std::vector<T> in(n);
    for (uint32_t k = 0; k < n; k++) in[k] = T(float(lo + (hi - lo) * k / (n - 1)));  // bf16: RNE on the host
    distributed::EnqueueWriteMeshBuffer(cq, in_buf, in, false);

    for (tt::CBIndex cb : {tt::CBIndex::c_0, tt::CBIndex::c_16})
        CreateCircularBuffer(program, core, CircularBufferConfig(2 * tile_bytes, {{cb, df}}).set_page_size(cb, tile_bytes));

    std::vector<uint32_t> reader_ct, writer_ct;
    TensorAccessorArgs(*in_buf).append_to(reader_ct);
    TensorAccessorArgs(*out_buf).append_to(writer_ct);
    // The stock dataflow kernels of custom_sfpi_smoothstep (found under the runtime root).
    auto reader = CreateKernel(program, OVERRIDE_KERNEL_PREFIX "custom_sfpi_smoothstep/kernels/dataflow/read_tiles.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_ct});
    auto writer = CreateKernel(program, OVERRIDE_KERNEL_PREFIX "custom_sfpi_smoothstep/kernels/dataflow/write_tile.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_ct});
    std::string up = fn;
    std::transform(up.begin(), up.end(), up.begin(), ::toupper);
    std::vector<UnpackToDestMode> u2d_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (u2d) u2d_modes[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
    auto compute = CreateKernel(program, "b08/sfpu_lab.cpp", core,
        ComputeConfig{.fp32_dest_acc_en = dst32, .unpack_to_dest_mode = u2d_modes, .math_approx_mode = approx,
                      .defines = {{"B08_FN_" + up, "1"}, {"B08_EXPT", expt ? "true" : "false"}, {"B08_FOLD", std::to_string(fold)}}});
    SetRuntimeArgs(program, reader, core, {in_buf->address(), n_tiles});
    SetRuntimeArgs(program, writer, core, {out_buf->address(), n_tiles});
    SetRuntimeArgs(program, compute, core, {n_tiles});

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    std::vector<T> out;
    distributed::EnqueueReadMeshBuffer(cq, out, out_buf, true);
    mesh_device->close();

    uint64_t hash = 1469598103934665603ull;  // FNV-1a over the output bits
    for (auto& v : out) { hash = (hash ^ std::bit_cast<uint32_t>(float(v))) * 1099511628211ull; }

    if (fn == "map") {  // face 0 of tile 0: which (vector, lane) wrote each cell
        std::printf("face 0 of tile 0, each cell = vector.lane (value = 32*vector + lane)\n");
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) { int v = int(float(out[r * 16 + c])); std::printf("%2d.%-2d ", v / 32, v % 32); }
            std::printf("\n");
        }
        bool same = true;  // the LLK runs the same face function on faces 1-3
        for (uint32_t k = 256; k < n; k++) same &= float(out[k]) == float(out[k % 256]);
        std::printf("faces 1..%u identical to face 0: %s\nresult_hash=%016llx\n", 4 * n_tiles - 1, same ? "yes" : "no", (unsigned long long)hash);
        return 0;
    }

    FILE* dump = env("B08_DUMP", "").empty() ? nullptr : std::fopen(env("B08_DUMP", "").c_str(), "w");
    double max_abs = 0, sum_abs = 0, max_rel = 0, x_at_max = 0;
    uint32_t ulp0 = 0, ulp1 = 0, ulp2 = 0; int64_t max_ulp = 0;
    for (uint32_t k = 0; k < n; k++) {
        const double x = float(in[k]), ref = reference(fn, x), got = float(out[k]);
        const double e = std::fabs(got - ref);
        if (e > max_abs) { max_abs = e; x_at_max = x; }
        sum_abs += e;
        if (std::fabs(ref) > 1e-6) max_rel = std::max(max_rel, e / std::fabs(ref));
        // distance from the correctly rounded (RNE) reference, in units of the output format's last place
        const int64_t d = std::llabs(ordinal(float(got), !fp32) - ordinal(float(T(float(ref))), !fp32));
        (d == 0 ? ulp0 : d == 1 ? ulp1 : ulp2)++;
        max_ulp = std::max(max_ulp, d);
        if (dump) std::fprintf(dump, "%.9g %.9g %.9g\n", x, ref, got);
    }
    if (dump) std::fclose(dump);
    std::printf("fn=%s fmt=%s dst32=%d u2d=%d approx=%d expt=%d fold=%d range=[%g,%g] n=%u\n", fn.c_str(),
                fp32 ? "fp32" : "bf16", dst32, u2d, approx, expt, fold, lo, hi, n);
    std::printf("max_abs_err=%.3g (at x=%.6g) mean_abs_err=%.3g max_rel_err=%.3g\n", max_abs, x_at_max, sum_abs / n, max_rel);
    std::printf("ulp vs correctly rounded: exact=%u 1ulp=%u >1ulp=%u max=%lld\nresult_hash=%016llx\n", ulp0, ulp1, ulp2,
                (long long)max_ulp, (unsigned long long)hash);
    return 0;
}

int main() {
    const std::string fn = env("B08_FN", "piecewise");
    const bool fp32 = env("B08_FMT", "bf16") == "fp32";
    const bool dst32 = std::atoi(env("B08_DST32", fp32 ? "1" : "0").c_str());
    const bool u2d = std::atoi(env("B08_U2D", "0").c_str());
    const bool approx = std::atoi(env("B08_APPROX", "0").c_str());
    const int expt = std::atoi(env("B08_EXPT", "0").c_str());
    const int fold = std::atoi(env("B08_FOLD", "1").c_str());
    double lo = -2, hi = 2;  // piecewise
    if (fn == "sin") { lo = -M_PI; hi = M_PI; }
    if (fn == "exp") { lo = -8; hi = 4; }
    if (fn == "recip") { lo = 0.5; hi = 8; }
    if (fn == "copy") { lo = 1; hi = 2; }
    lo = envf("B08_LO", lo); hi = envf("B08_HI", hi);
    const uint32_t n_tiles = std::atoi(env("B08_TILES", fn == "map" ? "1" : "4").c_str());
    return fp32 ? run<float>(fn, fp32, dst32, u2d, approx, expt, fold, lo, hi, n_tiles)
                : run<bfloat16>(fn, fp32, dst32, u2d, approx, expt, fold, lo, hi, n_tiles);
}
