#!/bin/bash
# Headset replay, part 2: after the VLM replay finishes, push the frames and time localisation on the Quest.
ADB="$HOME/android-sdk/platform-tools/adb -s HEADSET_SERIAL"
H=$(dirname "$(readlink -f "$0")"); R=/data/local/tmp/edb/ocr; OUT=$HOME/qwen/phone_bench/dvfs/revision/replay_loc.tsv
while pgrep -f "replay_vlm.py" > /dev/null; do sleep 20; done
echo "VLM replay finished $(date)"
$ADB shell "pkill -f '[l]lama-server-opencl'; mkdir -p $R/frames"
$ADB push "$H/replay_frames/." $R/frames/ | tail -1
$ADB push "$H/ocrreplay" "$H/replay_lines.txt" $R/ | tail -1
$ADB shell "chmod +x $R/ocrreplay; am broadcast -a com.oculus.vrpowermanager.prox_close > /dev/null"
sleep 30                                                        # let the push settle
echo "start localisation $(date)"
$ADB shell "cd $R && LD_LIBRARY_PATH=$R nohup taskset 3c ./ocrreplay PP-OCRv5_mobile_det.onnx latin_PP-OCRv5_mobile_rec.onnx \
  replay_lines.txt frames 4 > replay_loc.tsv 2> replay_loc.err &"
n=0
while true; do
  sleep 60
  $ADB shell "am broadcast -a com.oculus.vrpowermanager.prox_close > /dev/null"
  n=$($ADB shell "wc -l < $R/replay_loc.tsv" | tr -d '\r ')
  echo "$(date +%T) $n rows"
  $ADB shell "pgrep -f '[o]crreplay'" > /dev/null || break
done
$ADB pull $R/replay_loc.tsv "$OUT" | tail -1
$ADB shell "cat $R/replay_loc.err"
echo "done $(date)"
