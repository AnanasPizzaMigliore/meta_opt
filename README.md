# meta_opt

Code and data for "Reading the Fine Print on a Standalone XR Headset: Where an On-Device VLM Spends Its Time,
and What to Do with the Time Won Back".

A fine-tuned 0.8B vision-language model (Qwen3.5-0.8B) reads expiration dates on a Meta Quest 3S with
llama.cpp (OpenCL backend). This repository holds the runtime changes, the benchmark and replay harnesses,
the stopping rules, and the per-frame data behind every table and figure.

## Layout

| Folder | Contents |
|---|---|
| `patches/` | Patch against llama.cpp commit `8f9ae20` (task-vocabulary output head, server prefix-cache fix, vision-encoder changes). The tiled F32 kernel is re-enabled at run time with `GGML_OPENCL_A7X_F32_LM_BYPASS=0`. |
| `model/` | `add_task_vocab.py`: builds the 285-token task-vocabulary head into the GGUF model. |
| `bench/` | On-headset latency, energy and accuracy harnesses (`e2e_bench.py`, `energy_bench.py`, `server_bench.py`), the VLM replay (`replay_vlm.py`), Table 1 (`table1.py`), the battery trace config and the 40-crop latency sample. |
| `ocr/` | On-headset localisation timing: `ocrreplay.cpp` (ONNX Runtime, PP-OCRv5 mobile detector + latin recogniser), the line boxes it replays, and the run script. |
| `capture/` | The passthrough-camera capture app (`MainActivity.java`, manifest) and the recording script. |
| `policies/` | Localisation of real clips, VLM reads, the stopping rules (`reading_policy.py`, fixed before the held-out products were evaluated), the real-video evaluation (`real_eval.py`, Table 3) and the replay timing (`replay_time.py`). |
| `simulation/` | Simulated headset frames, OCR-only baseline, voting policies, Table 2 and Figure 4. |
| `figures/` | Figure 2 and the shared plot style. |
| `data/measurements/` | Raw measurements: latency (3 rounds), energy on battery, deployed accuracy on 544 crops, VLM replay reads, localisation replay timings, and the composed replay results. |
| `data/real/` | Ground truth for the 40 recorded products (`truth.json`, dates as printed, day first) and, per clip, the localisation output (`meta.json`: detector level, chosen lines, all recognised lines) and the VLM reads at the 1024- and 384-token caps. |

## Reproducing the main results

```
cd policies
python real_eval.py          # Table 3 outcomes on the 37 held-out products (p004-p040)
python replay_time.py ../data/measurements/replay_vlm.jsonl ../data/measurements/replay_loc.tsv
                             # Table 3 time column: per-frame headset costs summed along each rule's stopping path
python ../bench/table1.py    # Table 1 from the latency rounds
```

Scripts keep the absolute paths of the original setup (`/home/user/...`); adjust them to where the data is.

## Not included

- The 160 passthrough clips (about 3 GB) and the model weights are distributed separately.
- The test-set photos used for the simulated frames.

## Measurement notes

- Latency is the server's own prompt + decode time, excluding the adb transport.
- Localisation timing runs the detector on the full 1728x1728 frame and the recogniser on that frame's line
  crops, batched by 6; JPEG decoding is reported separately and not counted.
- Energy integrates a 100 ms battery current/voltage trace over each read, on battery, with the display on.
