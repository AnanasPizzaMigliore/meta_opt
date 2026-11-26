#!/bin/bash
# Headset runs without a rendering app (cable + tunnel only).
cd "$(dirname "$0")"
./run_live2.sh xcheck_opt xcheck opt
./run_live2.sh ocronly ocronly opt
./run_live2.sh xcheck_stock xcheck stock
./run_live2.sh gated_opt gated opt
echo ALLDONE
