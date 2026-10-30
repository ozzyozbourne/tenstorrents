# b09_cbpeek.py (lab B09): attach tt-exalens to a running, hung ttsim and read the circular-buffer
# counters of one core. Start the app with TT_METAL_SIMULATOR_SERVE_OVER_SOCKETS=1, then run
#   python -I b09_cbpeek.py /tmp/tt-umd-sim-server-<N> [core] [cb ...]      (defaults: 0,0 and CBs 0 1 16)
# On Wormhole the counters are stream scratch registers (tt-metal hw/inc/internal/tt-1xx/wormhole/stream_io_map.h):
# CB n uses stream n; tiles_received is register 4 and tiles_acked is register 3 of that stream, at
# NOC_OVERLAY_START_ADDR (0xFFB40000) + n * 0x1000 + reg * 4. Both are 16-bit counters.
import sys
from ttexalens.tt_exalens_init import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device

server_dir = sys.argv[1]
core = sys.argv[2] if len(sys.argv) > 2 else "0,0"
cbs = [int(c) for c in sys.argv[3:]] or [0, 1, 16]
ctx = init_ttexalens(simulation_directory=server_dir)
for cb in cbs:
    base = 0xFFB40000 + cb * 0x1000
    received = read_word_from_device(core, base + 4 * 4, context=ctx) & 0xFFFF
    acked = read_word_from_device(core, base + 3 * 4, context=ctx) & 0xFFFF
    print(f"cb {cb:2d}: tiles_received={received:3d} tiles_acked={acked:3d} in_flight={(received - acked) & 0xFFFF}")
