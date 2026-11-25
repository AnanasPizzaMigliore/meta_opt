#!/bin/bash
# Live decision runs on the Quest (cable + tunnel). Usage: run_live.sh <name> <rule> <config opt|stock> [app] [plan]
# Starting the foreground app from the shell can be refused by Horizon OS; then start it in the headset and pass "home".
#   app: package to keep rendering in the foreground during the run (e.g. com.meta.samples.discover), or "home"
set -e
NAME=$1; RULE=$2; CFG=$3; APP=${4:-home}; PLAN=${5:-live_plan.txt}
A="adb ${SERIAL:+-s $SERIAL}"
H=$(dirname "$(readlink -f "$0")"); R=/data/local/tmp/edb/ocr; OUT=$(dirname "$H")/data/real/live
mkdir -p $OUT
cd "$(dirname "$H")/bench"
if [ "$RULE" != "ocr2" ]; then
  if [ "$CFG" = opt ]; then CONF="+stop_if"; else CONF="orig"; fi
  python3 -c "
import e2e_bench as eb, server_bench as sb
m, env, extra, tf, stop, lv = eb.CONFIGS['$CONF']
eb.start(m, env, extra)
eb.read('0003.jpg', tf, stop)
print('server up: $CONF')"
fi
$A shell "am broadcast -a com.oculus.vrpowermanager.prox_close > /dev/null"
if [ "$APP" != home ]; then
  $A shell "am start -n $APP/com.unity3d.player.UnityPlayerActivity" || true; sleep 15
  $A shell "dumpsys activity activities | grep -m1 -E 'topResumedActivity|mResumedActivity'" || true
fi
$A push $H/$PLAN $R/live_plan_run.txt > /dev/null
echo "start $NAME $(date +%T)"
$A shell "cd $R && LD_LIBRARY_PATH=$R taskset 3c ./liveloop PP-OCRv5_mobile_det.onnx latin_PP-OCRv5_mobile_rec.onnx \
  live_plan_run.txt frames crops $RULE live_prefix_$CFG.json live_suffix_$CFG.json 8898 4 > live_$NAME.tsv 2> live_$NAME.err"
$A pull $R/live_$NAME.tsv $OUT/live_$NAME.tsv > /dev/null
$A shell "cat $R/live_$NAME.err | tail -3"
if [ "$APP" != home ]; then $A shell "am force-stop $APP"; fi
echo "done $NAME $(date +%T) clips $(grep -vc '^#' $OUT/live_$NAME.tsv)"
