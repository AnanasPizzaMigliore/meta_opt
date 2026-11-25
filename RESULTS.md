# Results not shown in the paper

The paper shows the main results; this file collects the remaining tables and numbers.
same study; scripts and raw data are in this repository (see README.md).

## Device, build and model details

- Meta Quest 3S: Snapdragon XR2 Gen 2, 4x2.36 GHz + 2x2.05 GHz CPU cores (dot-product, no i8mm), Adreno 740 GPU,
  Hexagon v69 NPU; Horizon OS build UP1A.231005.007.A1 (Android 14), GPU driver V@0837.0.9.
- Qwen3.5-0.8B: 18 Gated DeltaNet and 6 attention layers; 12-layer ViT, one visual token per 32x32 pixels.
  Q4_0 language weights, Q8_0 vision encoder. llama.cpp commit 8f9ae20 with `patches/`.
- Latency: 40 seeded crops, 3 rounds per configuration in alternating order; per-round medians differ by at most
  0.03 s and every answer is identical across rounds. Energy: 2 rounds per configuration in opposite order on
  battery; rounds agree within 12 % above idle and 5 % in total.

## Stock runtime (paper Sec. 3)

- NPU: opening a FastRPC session on the compute DSP fails with AEE_ERPC (0x200) from a shell and from an installed
  app; the compute-DSP device node is absent and the only FastRPC node is writable by the system user alone.
- Vulkan backend: wrong quantised matrix products on the Adreno 740 driver; garbled output.
- Prefill grows from 2.1 s (smallest crop) to 31.4 s (largest) with the encoder on the CPU; on the GPU with the tiled
  kernel the largest takes 11.5 s and no crop more than 14.4 s.
- Speculative decoding (own multi-token-prediction head, 10 images, task-vocabulary model): GPU 3 draft tokens,
  78 % accepted, decode 44.4 -> 68.9 ms/token; CPU 57.1 -> 45.0 ms/token.
- Frequency scaling: encoder follows the CPU clock, prefill the GPU clock, decode mostly the GPU clock; per-phase
  switching takes effect within 50 ms but reaches the same latency-energy points as the best fixed setting.

## Full latency table (paper Table 1 shows the endpoints)

| Configuration | Median (s) | p90 (s) | Speed-up within clock (95 % CI) | Exact (of 39) |
|---|---|---|---|---|
| Default clocks: baseline (ViT on CPU) | 4.87 | 33.6 | 1.00 | 20 |
| + tiled ViT kernel | 3.75 | 13.9 | 1.33 (1.26-1.50) | 20 |
| + task-vocabulary head | 3.12 | 13.3 | 1.53 (1.50-1.71) | 20 |
| + early stop | 2.23 | 12.3 | 2.15 (2.13-2.23) | 20 |
| Maximum clocks: baseline | 3.39 | 23.9 | 1.00 | 20 |
| All three changes | 1.51 | 8.4 | 2.23 (2.20-2.30) | 20 |
| + 384-token cap | 1.51 | 2.7 | 2.23 (2.21-2.31) | 20 |

Clocks alone: 1.43x; combined with the changes: 3.20x against the default-clock baseline. One answer changes,
from one wrong date to another, when the encoder moves to the GPU. Battery 36-38 C over the 100-minute run.
The 384-token cap costs 5.6 points of single-frame accuracy with the on-device detector (59.0 -> 53.4 %; 1.9 with
the server-grade detector) and raised invented dates on pilot no-date sides from 9 to 17 of 45 frames.

CPU decode with the task-vocabulary head and three speculative tokens: 70 -> 45 ms/token.

## Energy

Idle with the model loaded: 7.2 W (7.5-7.8 W at maximum clocks). Default clocks: energy a read adds above idle
8.7 -> 3.2 J (0.37x), total 43.7 -> 19.6 J (0.45x). Maximum clocks: 13.4 -> 4.9 J (0.36x), total 40.6 -> 16.2 J.
Maximum clocks lower total energy (16.2 vs 19.6 J) but raise the energy added above idle 1.5x (4.9 vs 3.2 J).

## Prompt-order pitfall

Placing the instructions before the image lets the server reuse 187 tokens; accuracy on dated products is
unchanged (74.2 vs 74.0 %), but the model (fine-tuned with the image first) invents a date for 24 of the 25
undated products instead of 7. We keep the image first.

## Localisation cost and accuracy

The server-grade detector takes 6.2 s per frame on the headset; the mobile detector 0.39 s on a 960 px frame and
1.24 s on a full 1728x1728 frame; recognition adds 59 ms per line (32 lines at the median), 3.3 s per real frame
(p90 5.1 s). With the mobile detector single-read accuracy drops from 63.4 to 58.4 %. The headset with all three
changes reads the same 544 crops at 58.6 % (304 of 519 dated) vs 58.4 % on the desktop backend; same answer on
477. PP-OCRv6 on the same 259 frames: tiny 1.25 s, small 3.50 s, v5 mobile 3.34 s (same session).

## Simulation (paper Sec. 5, one sentence)

545 products, seven simulated frames each (rotation +-3 deg, distance +-10 %, blur, sensor noise, brightness and
contrast +-8 %, JPEG quality 85-92), on-device detector, 1024-token cap. Exact counts abstentions as wrong;
precision is over answered dated inputs.

| Policy | Exact (%) | Coverage (%) | Precision on dated (%) | False dates | Frames read |
|---|---|---|---|---|---|
| One frame: OCR + date rules, no VLM | 29.9 | 37.8 | 79.2 | 3/175 | 1.0 |
| One frame: VLM | 59.0 | 100.0 | 59.0 | 70/175 | 1.0 |
| One frame: VLM confirmed by OCR | 27.3 | 28.0 | 97.2 | 0/175 | 1.0 |
| Vote over 3 | 64.0 | 100.0 | 64.0 | 10/25 | 3.0 |
| Until 2 agree | 64.8 | 99.4 | 65.2 | 6/25 | 2.7 |
| Vote over 5 | 65.4 | 100.0 | 65.4 | 6/25 | 5.0 |
| Answer if 3 of 5 agree | 60.0 | 78.8 | 76.1 | 4/25 | 5.0 |
| Until OCR confirms or 2 agree | 65.6 | 99.4 | 66.0 | 6/25 | 2.4 |
| Oracle: any of 7 frames correct | 80.2 | | | | |

Modelled time (cost models fitted to headset measurements; read time vs prompt tokens at maximum clocks,
R^2 = 0.875 over 78 reads): one frame 2.6 s median; until 2 agree 8.0 s (7.3 s with the 384-token cap; p90 27 ->
15 s); voting over 5 or 7 frames adds under one point for 2-3x the time. The cost model underestimates
localisation on real frames (2.4 vs 3.3 s). Figure: `figures/simulation_latency_accuracy.png`.
Repeated errors in simulation: 39 % of pairs of wrong reads of one image are identical (945 of 2425); "April
2027, no day" is emitted on 161 frames of 62 images.

## Full real-video table (paper Table 2 shows four rules)

PP-OCRv5 mobile, 37 held-out products, 111 dated clips, 37 no-date sides; decisions from desktop reads; time is
the serial sum of per-frame headset costs, median (p90).

| Rule | Correct | Wrong | No date | Abstain | Coverage % | False | Precision % (95 % CI) | VLM calls/clip | Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| VLM repeat (until 2 agree) | 63 | 25 | 20 | 3 | 79.3 | 16 | 60.6 (50-71) | 3.0 | 26.2 (46.8) |
| Gated VLM repeat (fixed policy) | 63 | 24 | 13 | 11 | 78.4 | 11 | 64.3 (53-75) | 2.8 | 22.2 (40.5) |
| Until OCR confirms or 2 agree | 65 | 24 | 20 | 2 | 80.2 | 16 | 61.9 (52-72) | 2.7 | 22.4 (45.1) |
| OCR cross-check, as specified | 53 | 0 | 0 | 58 | 47.7 | 0 | 100.0 | 5.1 | 43.9 (94.3) |
| OCR cross-check, OCR-date skip (post hoc) | 53 | 0 | 0 | 58 | 47.7 | 0 | 100.0 | 0.7 | 19.2 (33.8) |
| OCR first parsed date (post hoc) | 53 | 15 | 0 | 43 | 61.3 | 2 | 75.7 (62-87) | 0 | 15.6 (28.9) |
| OCR repeat, 2 frames (post hoc) | 49 | 3 | 0 | 59 | 46.8 | 0 | 94.2 (84-100) | 0 | 18.9 (30.2) |
| OCR repeat, 3 frames (post hoc) | 47 | 1 | | | 43.2 | 0 | 97.9 | 0 | 19.4 (30.4) |
| OCR repeat, 4 frames (post hoc) | 44 | 0 | | | 39.6 | 0 | 100.0 | 0 | 20.4 (31.6) |

Headset's own reads (871 of 1036 frames identical to the desktop): VLM repeat 62/26/13 false; gated 62/25/10;
cross-check 54/0/0, 19.5 (33.8) s. Paired, cross-check vs OCR repeat (2): 7 clips gained, 3 lost, net +4 (95 %
product-bootstrap CI -3 to +11). Posture (gated VLM repeat correct): bent arm 24, close 22, straight arm 17 of 37;
the cross-check abstains on 13 / 19 / 26. Repeated errors: 35 % of pairs of wrong reads within a clip identical
(895 of 2569) vs 4.8 % across products; 41 % of pairs of invented dates on no-date sides (493 of 1197).

## PP-OCRv6 (paper: one paragraph)

| OCR model | OCR date correct per dated frame | Gated VLM repeat precision | OCR repeat (2) | Cross-check correct / wrong / false (precision, 95 % CI) |
|---|---|---|---|---|
| PP-OCRv5 mobile | 38.5 % | 64.3 % | 94.2 % | 53 / 0 / 0 (100.0) |
| PP-OCRv6 tiny | 26.5 % | 64.1 % | 90.2 % | 45 / 1 / 1 (95.7, 85-100) |
| PP-OCRv6 small | 39.0 % | 63.4 % | 86.7 % | 51 / 3 / 0 (94.4, 87-100) |
| PP-OCRv6 medium | 43.9 % | 62.7 % | 85.3 % | 60 / 3 / 1 (93.8, 87-99) |

At no more errors than the cross-check, OCR repeat (2-4 frames) has 9 (v5), 11 (tiny) and 11 (medium) fewer
correct dates; with v6 small no OCR-repeat setting is as precise. Every confirmed v6 error is listed in
`data/real/clips_ppocrv6/ppocrv6_confirmed_errors.csv`.

## Packages with a second printed date

Six of the 37 products (`data/real/multidate_products.txt`). Precision on dated clips:

| OCR model | Cross-check, single-date (31) | Cross-check, multi-date (6) | Gated VLM repeat, single / multi |
|---|---|---|---|
| PP-OCRv5 mobile | 100 % (45/0) | 100 % (8/0) | 74.7 / 58.3 % |
| PP-OCRv6 tiny | 97.4 % (38/1) | 100 % (7/0) | 78.5 / 61.5 % |
| PP-OCRv6 small | 97.8 % (44/1) | 77.8 % (7/2) | 73.3 / 56.2 % |
| PP-OCRv6 medium | 98.1 % (51/1) | 81.8 % (9/2) | 75.3 / 52.9 % |

## Offloading estimate

A GPU server 16 ms away on campus Wi-Fi reads a median crop (44 kB) in 0.20 s: offloading the read takes about
3.5 s per frame plus transfer, nearly all on-headset localisation, against 7.0 s on device (median localisation
plus read per frame).

## Test-split screening

The split is image-level. Screening every test image against its nearest training images (DINOv2 embeddings, 189
candidate pairs checked by eye) found no shared product among the screened pairs, but 13 test images whose
product line, in another flavour, is in training; without them the headset reads 58.9 % (298/506; 95 % CI
54.3-63.0) against 58.6 % with them. Embedding screening can miss a product photographed very differently.

## Tuned OCR-only baseline (paper Sec. 5)

OCR-only rule OCR(k, tau): a date counts only if the recogniser's confidence in its line is >= tau; answer the
first date counted k times. Tuned on the simulated photos only (`policies/ocr_conf_tune.py`,
`data/measurements/ocr_conf_choice.json`, registered before any held-out evaluation): no setting reached the
OCR cross-check's development precision (best 88.9 % vs 96.0 %), so two fallbacks were registered: (a) the most
precise setting, k=2, tau=0.99; (b) the coverage-matched setting, k=3, tau=0.80. Held-out (37 products,
`policies/ocr_conf_eval.py`, output `data/measurements/ocr_conf_eval_2026-11-24.txt`):

| Rule | Correct | Wrong + false | Precision | Median time (s) |
|---|---|---|---|---|
| OCR cross-check (OCR-date skip) | 53 | 0 | 100 % | 19.2 |
| OCR-only (a) k=2, tau=0.99 | 28 | 0 | 100 % | 21.2 |
| OCR-only (b) k=3, tau=0.80 | 47 | 1 | 97.9 % | 19.4 |
| OCR repeat, k=2, no threshold | 49 | 3 | 94.2 % | 18.9 |

Paired by product (5000 resamples): cross-check minus (a) +25 correct (95 % CI +14..+36), same errors; minus (b)
+6 (-2..+14), one error fewer. The pilot products are uninformative here: OCR parses a date on 1 of their 63
dated frames. Confidence scores come from a rerun of the same OCR whose text is identical on all 1120 frames
(`data/real/ocr_scores_v5.json`).

## Live decision loop on the headset (paper Table 5)

`live/liveloop.cpp` runs on the Quest: sampled frames of each recorded clip arrive every 0.75 s, the loop
localises the newest frame (PP-OCRv5 mobile, ONNX Runtime, 4 CPU threads) and drops older ones, sends the crop
to the persistent llama-server asynchronously (reading overlaps with localising the next frame), and abstains
after 7 examined frames or at the clip's end. VLM replies are the headset's own; the date parsed from the OCR
text is taken from the offline pipeline (`live/make_live_plan.py`). Outputs in `data/real/live/`, summary with
`python policies/live_eval.py data/real/live/*.tsv`:

| Run | C / W / no-date / abstain / F | Cov. | Prec. | Decision median (p90) | To correct | To abstain | VLM calls |
|---|---|---|---|---|---|---|---|
| Gated VLM repeat | 63 / 15 / 9 / 24 / 4 | 70.3% | 76.8% | 17.2 (36.0) s | 17.0 (39.9) s | 17.6 (22.2) s | 2.26 |
| OCR repeat | 47 / 2 / 0 / 62 / 0 | 44.1% | 95.9% | 15.5 (20.0) s | 9.2 (17.8) s | 16.2 (19.9) s | 0 |
| OCR cross-check | 52 / 1 / 0 / 58 / 0 | 47.7% | 98.1% | 15.9 (21.2) s | 10.5 (19.7) s | 16.6 (21.3) s | 0.56 |
| Cross-check, stock read, 37 bent-arm clips | 23 / 1 / 0 / 13 / – | 64.9% | 95.8% | 17.9 (34.7) s | 18.4 (35.2) s | 17.8 (28.9) s | 0.92 |
| Cross-check, optimised read, same clips | 23 / 1 / 0 / 13 / – | 64.9% | 95.8% | 15.2 (20.9) s | 10.6 (19.7) s | 17.2 (20.9) s | 0.95 |

The cross-check's one wrong date (p013, side a) is the production date of a package that prints two dates.
Stock and optimised reads give the same answer on all 37 bent-arm clips. Per-frame times while both run:
localisation median 3.5–3.7 s (p90 5.5 s), VLM read median 2.5–2.8 s (p90 12 s, large crops); stock read median
8.1 s. Paired by product (10,000 resamples), the cross-check has 14 fewer wrong answers than gated VLM repeat
(95% CI [7, 22]) and 11 fewer correct ones ([3, 20]); against OCR repeat, +5 correct ([−2, +12]) and −1 wrong
([−6, +3]).
