#!/bin/bash
# Localisation timing on the Quest for PP-OCRv5 mobile (same session reference) and PP-OCRv6 tiny / small, on a
# fixed subset of the replay frames (every 4th frame of the replay list, 259 frames), same tool and threads.
ADB="$HOME/qwen/phone_bench/adbq -s HEADSET_SERIAL"
H=$(dirname "$(readlink -f "$0")"); R=/data/local/tmp/edb/ocr; OUT=$HOME/qwen/phone_bench/dvfs/revision
sub() { awk '/^F/{n++; keep=((n-1)%4==0)} keep' "$1"; }
sub $H/replay_lines.txt > $H/sub_v5.txt
$ADB push $H/sub_v5.txt $H/PP-OCRv6_tiny_det.onnx $H/PP-OCRv6_tiny_rec.onnx \
  $H/PP-OCRv6_small_det.onnx $H/PP-OCRv6_small_rec.onnx $H/ocrreplay $R/ | tail -1
n=$($ADB shell "ls $R/frames | wc -l" | tr -d '\r '); echo "frames on headset: $n"
[ "$n" = 1036 ] || { $ADB shell "mkdir -p $R/frames"; $ADB push "$H/replay_frames/." $R/frames/ | tail -1; }
$ADB shell "am broadcast -a com.oculus.vrpowermanager.prox_close > /dev/null; pkill -f '[l]lama-server-opencl'"
sleep 20
run() {  # name det rec list
  echo "start $1 $(date +%T)"
  $ADB shell "am broadcast -a com.oculus.vrpowermanager.prox_close > /dev/null"
  $ADB shell "cd $R && LD_LIBRARY_PATH=$R taskset 3c ./ocrreplay $2 $3 $4 frames 4 > loc_$1.tsv 2> loc_$1.err"
  $ADB pull $R/loc_$1.tsv $OUT/replay_loc_$1.tsv | tail -1; $ADB shell "cat $R/loc_$1.err"
  echo "done $1 $(date +%T) rows $(($(wc -l < $OUT/replay_loc_$1.tsv) - 1))"
  sleep 30
}
prep() { until grep -q "^done" $H/dump_lines_v6_$1.log 2>/dev/null; do sleep 20; done
  sub $H/replay_lines_v6_$1.txt > $H/sub_v6_$1.txt; $ADB push $H/sub_v6_$1.txt $R/ | tail -1; }
run v5sub PP-OCRv5_mobile_det.onnx latin_PP-OCRv5_mobile_rec.onnx sub_v5.txt
prep small; run v6small PP-OCRv6_small_det.onnx PP-OCRv6_small_rec.onnx sub_v6_small.txt
prep tiny; run v6tiny PP-OCRv6_tiny_det.onnx PP-OCRv6_tiny_rec.onnx sub_v6_tiny.txt
echo ALLDONE
