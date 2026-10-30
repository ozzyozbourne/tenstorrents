# Source this file before any lab:   source sim-labs/workbench/env.sh [wh|bh]
# It points tt-metal at the ttsim simulator and keeps all generated output out of the tt-metal tree.
# Every path below can be overridden by exporting the variable before sourcing.

ARCH=${1:-${ARCH:-wh}}
# Your built tt-metal and ttsim checkouts. Defaults: ~/tt-metal and ~/ttsim, falling back to the
# series' own orb layout (~/workspace/refs/...) when those do not exist.
if [ -z "$TT_METAL_HOME" ]; then
    TT_METAL_HOME=$HOME/tt-metal
    [ -d "$TT_METAL_HOME/tt_metal" ] || [ ! -d "$HOME/workspace/refs/tt-metal" ] || TT_METAL_HOME=$HOME/workspace/refs/tt-metal
fi
if [ -z "$TTSIM_HOME" ]; then
    TTSIM_HOME=$HOME/ttsim
    [ -f "$TTSIM_HOME/make.py" ] || [ ! -d "$HOME/workspace/refs/ttsim" ] || TTSIM_HOME=$HOME/workspace/refs/ttsim
fi
LAB_HOME=${LAB_HOME:-$HOME/ttlab}                                      # scratch space for lab output
mkdir -p "$LAB_HOME"
# Make the paths absolute: the script cds away at the end, so relative values would break.
TT_METAL_HOME=$(CDPATH= cd -- "$TT_METAL_HOME" 2>/dev/null && pwd || echo "$TT_METAL_HOME")
TTSIM_HOME=$(CDPATH= cd -- "$TTSIM_HOME" 2>/dev/null && pwd || echo "$TTSIM_HOME")
LAB_HOME=$(CDPATH= cd -- "$LAB_HOME" && pwd)
export TT_METAL_HOME TTSIM_HOME LAB_HOME

# The simulator: a libttsim.so that sits next to a soc_descriptor.yaml (tt-metal requires it).
export TT_METAL_SIMULATOR=${SIM:-$HOME/sim/$ARCH/libttsim_$ARCH.so}
[ -f "$(dirname "$TT_METAL_SIMULATOR")/soc_descriptor.yaml" ] ||
    echo "WARNING: no soc_descriptor.yaml next to $TT_METAL_SIMULATOR (runs will abort with YAML::BadFile)"
export TT_METAL_SLOW_DISPATCH_MODE=1          # recommended for ttsim (see README "Known Issues")
export TT_METAL_DISABLE_SFPLOADMACRO=1        # SFPLOADMACRO is UnsupportedFunctionality in ttsim

# Run from a lab directory, not from TT_METAL_HOME: kernel paths are searched in the
# current directory first, so running from TT_METAL_HOME would silently pick the stock kernels.
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME   # where firmware and system kernels live
# Your kernel overlay. A value you exported is kept, but one set by an earlier `source env.sh`
# follows the new LAB_HOME, so labs never share an overlay by accident.
if [ -z "$TT_METAL_KERNEL_PATH" ] || [ "$TT_METAL_KERNEL_PATH" = "$_TTLAB_KERNEL_PATH" ]; then
    TT_METAL_KERNEL_PATH=$LAB_HOME/kernels; _TTLAB_KERNEL_PATH=$TT_METAL_KERNEL_PATH
else
    _TTLAB_KERNEL_PATH=
fi
export TT_METAL_KERNEL_PATH
export TT_METAL_LOGS_PATH=$LAB_HOME/logs      # watcher / dprint / inspector output
export TT_METAL_PROFILER_DIR=$LAB_HOME/profiler
export EX=$TT_METAL_HOME/build_Release/programming_examples   # prebuilt example binaries
mkdir -p "$LAB_HOME/kernels" "$LAB_HOME/logs" "$LAB_HOME/run"
cd "$LAB_HOME/run"
echo "ttsim lab env: ARCH=$ARCH SIM=$TT_METAL_SIMULATOR cwd=$PWD"
