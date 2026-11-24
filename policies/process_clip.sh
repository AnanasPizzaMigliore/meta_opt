#!/bin/bash
# process_clip.sh <product number> <clip letter> [set name, default pilot]
# Pull one clip from the headset (if not pulled yet), localise the date in every 6th frame, read the crops with the VLM
# at the 1024- and 384-token caps (host GPU), write a contact sheet, and print a summary.
p=$(printf 'p%03d' "$1"); c=$2; set_=${3:-pilot}
A="$HOME/qwen/phone_bench/adbq -s ${QUEST:-HEADSET_IP:5555}"
P=$HOME/qwen/phone_bench/questcap_captures/$set_; H=$(dirname "$(readlink -f "$0")")
mkdir -p "$P/$p" "$P/loc"
if [ ! -f "$P/$p/$c/frames.tsv" ]; then
  (cd "$P/$p" && $A pull "/sdcard/Android/data/org.example.questcap/files/captures/$p/$c" . | tail -1) || exit 1
fi
L=$P/loc/${p}_${c}_rot0
[ -f "$L/meta.json" ] || ~/miniconda3/envs/ocr/bin/python "$H/loc_clip.py" "$P/$p/$c" "$L" --every 6 > "$L.loc.log" 2>&1
for cap in 1024 384; do
  [ -f "$L/reads_cap$cap.json" ] || ~/miniconda3/envs/llm/bin/python "$H/read_crops.py" "$L/x2" "$L/reads_cap$cap.json" --cap $cap > "$L/reads_cap$cap.log" 2>&1
done
~/miniconda3/envs/ocr/bin/python "$H/summarise_clip.py" "$P/$p/$c" "$L"
