# meta_opt

Code and data for "Reading the Fine Print on a Standalone XR Headset: On-Device VLM Latency and Reliability".

A fine-tuned 0.8B vision-language model (Qwen3.5-0.8B) reads expiration dates on a Meta Quest 3S with
llama.cpp (OpenCL backend). This repository holds the runtime changes, the benchmark and replay harnesses,
the stopping rules, and the per-frame data behind every table and figure.

Results that the paper summarises in a sentence (full tables, simulation, energy, PP-OCRv6, multi-date packages,
offloading, split screening) are in [RESULTS.md](RESULTS.md).

## Layout

| Folder | Contents |
|---|---|
| `patches/` | Patch against llama.cpp commit `8f9ae20` (task-vocabulary output head, server prefix-cache fix, vision-encoder changes). The tiled F32 kernel is re-enabled at run time with `GGML_OPENCL_A7X_F32_LM_BYPASS=0`. |
| `model/` | `add_task_vocab.py`: builds the 285-token task-vocabulary head into the GGUF model. |
| `bench/` | On-headset latency, energy and accuracy harnesses (`e2e_bench.py`, `energy_bench.py`, `server_bench.py`), the VLM replay (`replay_vlm.py`), Table 1 (`table1.py`), the battery trace config and the 40-crop latency sample. |
| `ocr/` | On-headset localisation timing: `ocrreplay.cpp` (ONNX Runtime, PP-OCRv5 mobile detector + latin recogniser), the line boxes it replays, and the run script. |
| `ocr/` (PP-OCRv6) | `dump_lines_v6.py` and `run_replay_loc_v6.sh`: the same localisation timing with PP-OCRv6 tiny and small (ONNX exports of the official models) and a same-session PP-OCRv5 reference, on every 4th replay frame (259 frames); results in `data/measurements/replay_loc_{v5sub,v6small,v6tiny}.tsv`. |
| `capture/` | The passthrough-camera capture app (`MainActivity.java`, manifest) and the recording script. |
| `policies/` | Localisation of real clips, VLM reads, the stopping rules (`reading_policy.py`, fixed before the held-out products were evaluated), the real-video evaluation (`real_eval.py`, Table 3) and the replay timing (`replay_time.py`). |
| `simulation/` | Simulated headset frames, OCR-only baseline, voting policies, Table 2 and Figure 4. |
| `figures/` | Figure 2 and the shared plot style. |
| `data/measurements/` | Raw measurements: latency (3 rounds), energy on battery, deployed accuracy on 544 crops, VLM replay reads, localisation replay timings, and the composed replay results. |
| `data/real/clips_ppocrv6/` | The same per-clip outputs with PP-OCRv6 tiny / small / medium in place of PP-OCRv5 for localisation and OCR (robustness check), and `ppocrv6_confirmed_errors.csv`: every date the VLM-confirmed-by-OCR rule got wrong under PP-OCRv6, with the frame, the OCR lines and the audited cause. |
| `data/real/` | Ground truth for the 40 recorded products (`truth.json`, dates as printed, day first) and, per clip, the localisation output (`meta.json`: detector level, chosen lines, all recognised lines) and the VLM reads at the 1024- and 384-token caps. |

## Reproducing the main results

Run from the repository root with Python 3 and numpy; the commands below read only files in `data/`.

```
python bench/table1.py              # Table 1: latency of the three changes (3 rounds x 40 crops x 7 configurations)
cd policies
python real_eval.py                 # Table 3 outcomes on the 37 held-out products (p004-p040), with bootstrap CIs
python replay_time.py               # Table 3 time column, all seven rows including "read only frames with an OCR
                                    # date", decisions from desktop reads and from headset reads, and the paired
                                    # comparison with OCR only (net correct clips, product bootstrap interval)
python eval_ppocrv6.py tiny small medium   # Table 3's PP-OCRv6 rows: every rule with each PP-OCRv6 tier
```

`replay_time.py` sums per-frame costs measured on the headset (`data/measurements/replay_loc.tsv`: localisation;
`replay_vlm.jsonl`: VLM reads) along each rule's stopping path. The other scripts (capture, benchmarks on the
headset, simulation) keep the absolute paths of the original setup (`/home/user/...`) and need the headset, the
model files or the test-set photos.

## Not included

- The 160 passthrough clips (about 10 GB of frames) and the model weights are distributed separately.
- The test-set photos used for the simulated frames.

## Measurement notes

- Latency is the server's own prompt + decode time, excluding the adb transport.
- Localisation timing runs the detector on the full 1728x1728 frame and the recogniser on that frame's line
  crops, batched by 6; JPEG decoding is reported separately and not counted.
- Energy integrates a 100 ms battery current/voltage trace over each read, on battery, with the display on.
