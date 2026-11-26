#!/bin/bash
# Live decision runs on the Quest with on-device localisation (liveloop2). Usage:
#   run_live2.sh <name> <rule xcheck|gated|ocronly> <config opt|stock> [render: none|<package>] [plan]
# render <package>: that app must already be in the foreground (Horizon OS refuses shell launches); the run aborts
# if it is not, and checks it again at the end.
set -e
NAME=$1; RULE=$2; CFG=$3; RENDER=${4:-none}; PLAN=${5:-live_plan.txt}
SERIAL=${SERIAL:?set SERIAL to the headset serial}; export ANDROID_SERIAL=$SERIAL
A="$HOME/qwen/phone_bench/adbq -s $SERIAL"
H=$(dirname "$(readlink -f "$0")"); R=/data/local/tmp/edb/ocr; OUT=$HOME/qwen/phone_bench/dvfs/revision/live2
mkdir -p $OUT
fg() { $A shell "dumpsys activity activities | grep -m1 -E 'topResumedActivity|mResumedActivity'" || true; }
if [ "$RENDER" != none ]; then
  if ! fg | grep -q "$RENDER"; then echo "ABORT: $RENDER is not in the foreground: $(fg)"; exit 3; fi
fi
cd $HOME/qwen/phone_bench/dvfs
if [ "$RULE" != "ocronly" ]; then
  if [ "$CFG" = opt ]; then CONF="+stop_if"; else CONF="orig"; fi
  python3 -c "
import e2e_bench as eb, server_bench as sb
m, env, extra, tf, stop, lv = eb.CONFIGS['$CONF']
eb.start(m, env, extra)
eb.read('0003.jpg', tf, stop)
print('server up: $CONF')"
fi
if [ "$RENDER" = none ]; then $A shell "am broadcast -a com.oculus.vrpowermanager.prox_close > /dev/null"; fi
$A push $H/liveloop2 $H/latin_dict.txt $R/ > /dev/null
$A push $H/$PLAN $R/live_plan_run.txt > /dev/null
echo "start $NAME $(date +%T) fg: $(fg)"
$A shell "cd $R && LD_LIBRARY_PATH=$R taskset 3c ./liveloop2 PP-OCRv5_mobile_det.onnx latin_PP-OCRv5_mobile_rec.onnx latin_dict.txt \
  live_plan_run.txt frames $RULE live_prefix_$CFG.json live_suffix_$CFG.json 8898 4 > live2_$NAME.tsv 2> live2_$NAME.err" || echo "liveloop2 exit $?"
$A pull $R/live2_$NAME.tsv $OUT/live2_$NAME.tsv > /dev/null
$A shell "tail -3 $R/live2_$NAME.err"
echo "done $NAME $(date +%T) rows $(grep -vc '^#' $OUT/live2_$NAME.tsv) fg at end: $(fg)"
