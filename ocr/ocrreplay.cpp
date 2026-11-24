// Localisation replay on the headset: for each recorded frame, run the PP-OCRv5 mobile detector on the real frame
// (same input size as the accuracy runs), a DB-style post-processing pass (binarise at 0.3, connected components,
// mean score per component), then the latin recogniser on the frame's real line crops (boxes from the same
// pipeline on the server), batched by 6 in aspect-ratio order with CTC greedy decoding, as PaddleOCR does.
//
//   ocrreplay DET.onnx REC.onnx LIST FRAME_DIR THREADS
// LIST:  F <frame.jpg> <W> <H>  followed by  L x0 y0 x1 y1  lines.   Prints one TSV row per frame (times in ms).
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); }

struct Frame { std::string name; int W, H; std::vector<std::array<int, 4>> boxes; };

// bilinear resize of an RGB8 region into a CHW float tensor slice with per-channel normalisation (BGR order)
static void resize_norm(const unsigned char *img, int iw, int x0, int y0, int w, int h, float *dst, int ow, int oh,
                        int stride_w, const float mean[3], const float stdv[3], float scale) {
    for (int y = 0; y < oh; y++) {
        float sy = (y + 0.5f) * h / oh - 0.5f; int y1 = std::max(0, (int) sy); int y2 = std::min(h - 1, y1 + 1);
        float fy = std::max(0.f, sy - y1);
        for (int x = 0; x < ow; x++) {
            float sx = (x + 0.5f) * w / ow - 0.5f; int x1 = std::max(0, (int) sx); int x2 = std::min(w - 1, x1 + 1);
            float fx = std::max(0.f, sx - x1);
            for (int c = 0; c < 3; c++) {
                int cc = 2 - c;  // BGR
                auto P = [&](int yy, int xx) { return (float) img[((y0 + yy) * iw + (x0 + xx)) * 3 + cc]; };
                float v = (1 - fy) * ((1 - fx) * P(y1, x1) + fx * P(y1, x2)) + fy * ((1 - fx) * P(y2, x1) + fx * P(y2, x2));
                dst[(size_t) c * oh * stride_w + (size_t) y * stride_w + x] = (v * scale - mean[c]) / stdv[c];
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ocrreplay DET REC LIST FRAME_DIR THREADS\n"); return 2; }
    int threads = atoi(argv[5]);
    std::vector<Frame> frames;
    { std::ifstream in(argv[3]); std::string line;
      while (std::getline(in, line)) {
          std::istringstream ss(line); std::string tag; ss >> tag;
          if (tag == "F") { Frame f; ss >> f.name >> f.W >> f.H; frames.push_back(f); }
          else if (tag == "L") { std::array<int, 4> b; ss >> b[0] >> b[1] >> b[2] >> b[3]; frames.back().boxes.push_back(b); }
      } }
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "replay");
    Ort::SessionOptions so; so.SetIntraOpNumThreads(threads); so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session det(env, argv[1], so), rec(env, argv[2], so);
    Ort::AllocatorWithDefaultOptions alloc;
    auto din = det.GetInputNameAllocated(0, alloc), dout = det.GetOutputNameAllocated(0, alloc);
    auto rin = rec.GetInputNameAllocated(0, alloc), rout = rec.GetOutputNameAllocated(0, alloc);
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const float dmean[3] = {0.485f, 0.456f, 0.406f}, dstd[3] = {0.229f, 0.224f, 0.225f};
    const float rmean[3] = {0.5f, 0.5f, 0.5f}, rstd[3] = {0.5f, 0.5f, 0.5f};
    printf("frame\tdecode_ms\tdet_pre_ms\tdet_ms\tpost_ms\tcomponents\tn_lines\trec_ms\tloc_ms\n");
    for (size_t k = 0; k <= frames.size(); k++) {
        const Frame &f = frames[k == 0 ? 0 : k - 1];          // k == 0: warm-up on the first frame, not printed
        std::string path = std::string(argv[4]) + "/" + f.name;
        auto t0 = clk::now();
        int iw, ih, ch; unsigned char *img = stbi_load(path.c_str(), &iw, &ih, &ch, 3);
        if (!img) { fprintf(stderr, "cannot read %s\n", path.c_str()); return 1; }
        double t_dec = ms(t0);
        // detector: resize to multiples of 32 (frames are already 1728x1728, as in the accuracy runs)
        auto t1 = clk::now();
        int W = std::max(32, (int) std::round(iw / 32.0) * 32), H = std::max(32, (int) std::round(ih / 32.0) * 32);
        std::vector<float> x((size_t) 3 * H * W);
        resize_norm(img, iw, 0, 0, iw, ih, x.data(), W, H, W, dmean, dstd, 1.f / 255);
        std::array<int64_t, 4> dshape{1, 3, H, W};
        Ort::Value dt = Ort::Value::CreateTensor<float>(mem, x.data(), x.size(), dshape.data(), 4);
        double t_dpre = ms(t1);
        auto t2 = clk::now();
        const char *di[] = {din.get()}, *dO[] = {dout.get()};
        auto dres = det.Run(Ort::RunOptions{nullptr}, di, &dt, 1, dO, 1);
        double t_det = ms(t2);
        // DB-style post-processing: binarise, connected components (4-neighbour), keep those with mean score >= 0.6
        auto t3 = clk::now();
        const float *pm = dres[0].GetTensorData<float>();
        std::vector<int> lab((size_t) H * W, 0), stack; int ncomp = 0, kept = 0;
        for (int i = 0; i < H * W; i++) {
            if (pm[i] <= 0.3f || lab[i]) continue;
            ++ncomp; double s = 0; int n = 0; int bx0 = W, by0 = H, bx1 = 0, by1 = 0;
            stack.assign(1, i); lab[i] = ncomp;
            while (!stack.empty()) {
                int p = stack.back(); stack.pop_back(); s += pm[p]; n++;
                int py = p / W, px = p % W; bx0 = std::min(bx0, px); bx1 = std::max(bx1, px); by0 = std::min(by0, py); by1 = std::max(by1, py);
                const int nb[4] = {p - 1, p + 1, p - W, p + W};
                for (int q = 0; q < 4; q++) {
                    int r = nb[q];
                    if ((q == 0 && px == 0) || (q == 1 && px == W - 1) || r < 0 || r >= H * W) continue;
                    if (pm[r] > 0.3f && !lab[r]) { lab[r] = ncomp; stack.push_back(r); }
                }
            }
            if (s / n >= 0.6 && bx1 - bx0 >= 3 && by1 - by0 >= 3) kept++;
        }
        double t_post = ms(t3);
        // recogniser: real line crops, sorted by aspect ratio, batches of 6, height 48, width from the widest line
        auto t4 = clk::now();
        std::vector<std::array<int, 4>> bs = f.boxes;
        for (auto &b : bs) { b[0] = std::clamp(b[0], 0, iw - 1); b[2] = std::clamp(b[2], b[0] + 1, iw);
                              b[1] = std::clamp(b[1], 0, ih - 1); b[3] = std::clamp(b[3], b[1] + 1, ih); }
        std::sort(bs.begin(), bs.end(), [](auto &a, auto &b) {
            return (float) (a[2] - a[0]) / (a[3] - a[1]) < (float) (b[2] - b[0]) / (b[3] - b[1]); });
        long checksum = 0;
        for (size_t i = 0; i < bs.size(); i += 6) {
            size_t n = std::min<size_t>(6, bs.size() - i);
            float maxr = 320.f / 48;
            for (size_t j = 0; j < n; j++) maxr = std::max(maxr, (float) (bs[i + j][2] - bs[i + j][0]) / (bs[i + j][3] - bs[i + j][1]));
            int RW = (int) (48 * maxr);
            std::vector<float> r((size_t) n * 3 * 48 * RW, 0.f);
            for (size_t j = 0; j < n; j++) {
                auto &b = bs[i + j]; int w = b[2] - b[0], h = b[3] - b[1];
                int ow = std::min(RW, (int) std::ceil(48.f * w / h));
                resize_norm(img, iw, b[0], b[1], w, h, r.data() + j * 3 * 48 * RW, ow, 48, RW, rmean, rstd, 1.f / 255);
            }
            std::array<int64_t, 4> rshape{(int64_t) n, 3, 48, RW};
            Ort::Value rt = Ort::Value::CreateTensor<float>(mem, r.data(), r.size(), rshape.data(), 4);
            const char *ri[] = {rin.get()}, *ro[] = {rout.get()};
            auto rres = rec.Run(Ort::RunOptions{nullptr}, ri, &rt, 1, ro, 1);
            auto sh = rres[0].GetTensorTypeAndShapeInfo().GetShape();   // n x T x C
            const float *o = rres[0].GetTensorData<float>();
            for (int64_t a = 0; a < sh[0] * sh[1]; a++) {                 // CTC greedy decoding
                const float *row = o + a * sh[2];
                checksum += std::max_element(row, row + sh[2]) - row;
            }
        }
        double t_rec = ms(t4);
        stbi_image_free(img);
        if (k == 0) continue;
        printf("%s\t%.1f\t%.1f\t%.1f\t%.1f\t%d\t%zu\t%.1f\t%.1f\n", f.name.c_str(), t_dec, t_dpre, t_det, t_post, kept,
               bs.size(), t_rec, t_dpre + t_det + t_post + t_rec);
        fflush(stdout);
        (void) checksum;
    }
    return 0;
}
