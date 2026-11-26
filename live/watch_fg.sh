#!/bin/bash
# Stops liveloop2 if com.meta.samples.discover leaves the foreground on two checks in a row (30 s apart).
A="adb -s ${SERIAL:?set SERIAL to the headset serial}"; LOG=$1; miss=0
until grep -qE "^done|ABORT" $LOG; do
  sleep 30
  if $A shell "dumpsys activity activities | grep -m1 topResumedActivity" 2>/dev/null | grep -q com.meta.samples.discover; then miss=0
  else miss=$((miss+1)); echo "$(date +%T) not in front ($miss)"; fi
  if [ $miss -ge 2 ]; then P=$($A shell pidof liveloop2); [ -n "$P" ] && $A shell "kill $P"; echo "$(date +%T) WATCHDOG STOPPED RUN"; exit 1; fi
done
echo "$(date +%T) run finished with Discover in front"
