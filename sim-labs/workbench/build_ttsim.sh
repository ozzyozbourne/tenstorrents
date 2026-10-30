#!/bin/bash
# Make an instrumentable copy of ttsim, build it, and package it so tt-metal can load it.
#   build_ttsim.sh <name> [wh|bh|wh_x2|...] [release|debug]   (multichip configs: see src/rules.py CONFIGS)
# Creates $LAB_HOME/ttsim-<name>/ (a source copy you can edit) on first use, rebuilds it
# (about 4 s), and prints the export line for TT_METAL_SIMULATOR.
set -e
name=${1:?usage: build_ttsim.sh <name> [wh|bh] [release|debug]}
arch=${2:-wh}; flavor=${3:-release}
: "${TTSIM_HOME:?source env.sh first}" "${TT_METAL_HOME:?source env.sh first}" "${LAB_HOME:?source env.sh first}"
src=$LAB_HOME/ttsim-$name
if [ ! -d "$src" ]; then
    mkdir -p "$src"
    (cd "$TTSIM_HOME" && tar --exclude=./src/_out --exclude=./.git -cf - .) | (cd "$src" && tar -xf -)
    echo "copied ttsim source to $src (edit files in $src/src/)"
fi
(cd "$src" && ./make.py "src/_out/${flavor}_${arch}/libttsim.so" >/dev/null)
out=$LAB_HOME/sim-$name-$arch-$flavor
mkdir -p "$out"
cp "$src/src/_out/${flavor}_${arch}/libttsim.so" "$out/libttsim.so"
case $arch in
  wh|wh_x*) cp "$TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml" "$out/soc_descriptor.yaml" ;;
  bh|bh_x*) cp "$TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml" "$out/soc_descriptor.yaml" ;;
esac
echo "export TT_METAL_SIMULATOR=$out/libttsim.so"
