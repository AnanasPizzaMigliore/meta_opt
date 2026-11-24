#!/bin/bash
# Timed capture sequence, driven from the server (no controllers needed).
#   run_sequence.sh FIRST LAST [GAP_CLIP_S] [GAP_PRODUCT_S]
# For each product FIRST..LAST: clips a (bent arm), b (close), d (straight arm), c (side without a date). Each clip: warning beep, 2 s, 12 s recording, double beep.
# GAP_CLIP_S (default 8) to reposition between clips, GAP_PRODUCT_S (default 20) to change product.
A="$HOME/qwen/phone_bench/adbq -s ${QUEST:-HEADSET_IP:5555}"
first=${1:?first product}; last=${2:?last product}; gc=${3:-8}; gp=${4:-20}
for p in $(seq "$first" "$last"); do
  for c in ${CLIPS:-a b d c}; do
    echo "$(date +%T) product $p clip $c"
    $A shell am start -n org.example.questcap/.MainActivity --ei product "$p" --es clip "$c" --ez record true > /dev/null || exit 1
    sleep $((2 + 12 + 2))
    n=$($A shell "ls /sdcard/Android/data/org.example.questcap/files/captures/$(printf 'p%03d' "$p")/$c 2>/dev/null | wc -l")
    echo "   saved $n files"
    sleep "$gc"
  done
  [ "$p" != "$last" ] && sleep "$gp"
done
