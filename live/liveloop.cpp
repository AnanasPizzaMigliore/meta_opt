// Live decision loop on the headset. For each recorded clip, sampled frames "arrive" at their recorded cadence
// (0.75 s apart); the loop always takes the latest frame that has arrived and drops older ones, runs OCR
// localisation on it (PP-OCRv5 mobile detector + DB-style post-processing + recogniser on the frame's line boxes,
// ONNX Runtime, CPU), and, when the stopping rule needs it, sends the frame's crop to the persistent llama-server
// (GPU). The VLM call runs asynchronously, so OCR of the next frame overlaps with it. The rule decides from the
// VLM's actual reply on the headset; the OCR text and the date parsed from it (and the detector's gate level) come
// from the same OCR pipeline run on the server, because the recogniser's text decoding and the date rules are not
// ported to the headset.
//
//   liveloop DET REC PLAN FRAME_DIR CROP_DIR RULE PREFIX SUFFIX PORT THREADS [CADENCE_MS=750] [BUDGET=7]
// RULE: xcheck (VLM read only on frames with an OCR date; answer when VLM date == OCR date)
//       gated  (VLM read only on date/digit frames; answer when a valid date (or "no date") is read twice)
//       ocr2   (no VLM; answer when the same OCR date is seen on two frames)
// PLAN: "C clip n" then n x ("F frame level ocrdate nlines" + nlines x "L x0 y0 x1 y1"). Output: one TSV row per
// clip and one per examined frame (prefixed with '#').
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <onnxruntime_cxx_api.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <future>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); }

struct Frame { std::string name, level, ocr; std::vector<std::array<int, 4>> boxes; };
struct Clip { std::string id; std::vector<Frame> frames; };

static std::string slurp(const std::string &p) { std::ifstream f(p, std::ios::binary); std::stringstream s; s << f.rdbuf(); return s.str(); }

static std::string b64(const std::string &in) {
    static const char *t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out; out.reserve((in.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        unsigned v = (unsigned char) in[i] << 16 | (unsigned char) in[i + 1] << 8 | (unsigned char) in[i + 2];
        out += t[v >> 18]; out += t[(v >> 12) & 63]; out += t[(v >> 6) & 63]; out += t[v & 63];
    }
    if (i + 1 == in.size()) { unsigned v = (unsigned char) in[i] << 16; out += t[v >> 18]; out += t[(v >> 12) & 63]; out += "=="; }
    else if (i + 2 == in.size()) { unsigned v = (unsigned char) in[i] << 16 | (unsigned char) in[i + 1] << 8;
        out += t[v >> 18]; out += t[(v >> 12) & 63]; out += t[(v >> 6) & 63]; out += '='; }
    return out;
}

// POST to 127.0.0.1:port/v1/chat/completions, return body
static std::string post(int port, const std::string &body) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port); inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    if (connect(s, (sockaddr *) &a, sizeof a) != 0) { close(s); return ""; }
    std::string req = "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\n"
                      "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    size_t sent = 0;
    while (sent < req.size()) { ssize_t n = send(s, req.data() + sent, req.size() - sent, 0); if (n <= 0) break; sent += n; }
    std::string resp; char buf[65536]; ssize_t n;
    while ((n = recv(s, buf, sizeof buf, 0)) > 0) resp.append(buf, n);
    close(s);
    return resp;
}

static std::string parse_vlm(const std::string &resp) {   // -> "YYYY-MM-DD" with ?? fields, or NONE
    static const std::regex r("reads (\\d{4}|\\?{4})-(\\d{2}|\\?\\?)-(\\d{2}|\\?\\?)");
    std::smatch m;
    if (std::regex_search(resp, m, r)) return m[1].str() + "-" + m[2].str() + "-" + m[3].str();
    return "NONE";
}

static bool valid(const std::string &t) {   // calendar check of the gated rule; unknown fields allowed
    if (t == "NONE") return true;
    int y = t.substr(0, 4) == "????" ? 2000 : std::stoi(t.substr(0, 4));
    std::string ms = t.substr(5, 2), ds = t.substr(8, 2);
    int m = ms == "??" ? -1 : std::stoi(ms), d = ds == "??" ? -1 : std::stoi(ds);
    if (m != -1 && (m < 1 || m > 12)) return false;
    if (d != -1) {
        int last = 31;
        if (m != -1) { static const int L[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
            last = L[m - 1] + (m == 2 && ((y % 4 == 0 && y % 100) || y % 400 == 0)); }
        if (d < 1 || d > last) return false;
    }
    return true;
}

struct OCR {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "live"};
    Ort::Session *det, *rec;
    Ort::AllocatorWithDefaultOptions alloc;
    std::string din, dout, rin, rout;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    OCR(const char *d, const char *r, int threads) {
        Ort::SessionOptions so; so.SetIntraOpNumThreads(threads); so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        det = new Ort::Session(env, d, so); rec = new Ort::Session(env, r, so);
        din = det->GetInputNameAllocated(0, alloc).get(); dout = det->GetOutputNameAllocated(0, alloc).get();
        rin = rec->GetInputNameAllocated(0, alloc).get(); rout = rec->GetOutputNameAllocated(0, alloc).get();
    }
    static void resize_norm(const unsigned char *img, int iw, int x0, int y0, int w, int h, float *dst, int ow, int oh,
                            int stride, const float mean[3], const float sd[3]) {
        for (int y = 0; y < oh; y++) {
            float sy = (y + 0.5f) * h / oh - 0.5f; int y1 = std::max(0, (int) sy), y2 = std::min(h - 1, y1 + 1); float fy = std::max(0.f, sy - y1);
            for (int x = 0; x < ow; x++) {
                float sx = (x + 0.5f) * w / ow - 0.5f; int x1 = std::max(0, (int) sx), x2 = std::min(w - 1, x1 + 1); float fx = std::max(0.f, sx - x1);
                for (int c = 0; c < 3; c++) {
                    auto P = [&](int yy, int xx) { return (float) img[((y0 + yy) * iw + (x0 + xx)) * 3 + (2 - c)]; };
                    float v = (1 - fy) * ((1 - fx) * P(y1, x1) + fx * P(y1, x2)) + fy * ((1 - fx) * P(y2, x1) + fx * P(y2, x2));
                    dst[(size_t) c * oh * stride + (size_t) y * stride + x] = (v / 255.f - mean[c]) / sd[c];
                }
            }
        }
    }
    double run(const std::string &path, const std::vector<std::array<int, 4>> &boxes) {   // returns ms
        auto t0 = clk::now();
        int iw, ih, ch; unsigned char *img = stbi_load(path.c_str(), &iw, &ih, &ch, 3);
        if (!img) { fprintf(stderr, "cannot read %s\n", path.c_str()); return -1; }
        const float dm[3] = {0.485f, 0.456f, 0.406f}, ds[3] = {0.229f, 0.224f, 0.225f}, rm[3] = {.5f, .5f, .5f}, rs[3] = {.5f, .5f, .5f};
        int W = std::max(32, (int) std::round(iw / 32.0) * 32), H = std::max(32, (int) std::round(ih / 32.0) * 32);
        std::vector<float> x((size_t) 3 * H * W);
        resize_norm(img, iw, 0, 0, iw, ih, x.data(), W, H, W, dm, ds);
        std::array<int64_t, 4> sh{1, 3, H, W};
        Ort::Value t = Ort::Value::CreateTensor<float>(mem, x.data(), x.size(), sh.data(), 4);
        const char *di[] = {din.c_str()}, *dO[] = {dout.c_str()};
        auto res = det->Run(Ort::RunOptions{nullptr}, di, &t, 1, dO, 1);
        const float *pm = res[0].GetTensorData<float>();
        std::vector<int> lab((size_t) H * W, 0), st; int nc = 0;
        for (int i = 0; i < H * W; i++) {
            if (pm[i] <= 0.3f || lab[i]) continue;
            ++nc; st.assign(1, i); lab[i] = nc;
            while (!st.empty()) { int p = st.back(); st.pop_back(); int px = p % W;
                const int nb[4] = {p - 1, p + 1, p - W, p + W};
                for (int q = 0; q < 4; q++) { int r = nb[q];
                    if ((q == 0 && px == 0) || (q == 1 && px == W - 1) || r < 0 || r >= H * W) continue;
                    if (pm[r] > 0.3f && !lab[r]) { lab[r] = nc; st.push_back(r); } } }
        }
        auto bs = boxes;
        for (auto &b : bs) { b[0] = std::clamp(b[0], 0, iw - 1); b[2] = std::clamp(b[2], b[0] + 1, iw);
                             b[1] = std::clamp(b[1], 0, ih - 1); b[3] = std::clamp(b[3], b[1] + 1, ih); }
        std::sort(bs.begin(), bs.end(), [](auto &a, auto &b) { return (float) (a[2] - a[0]) / (a[3] - a[1]) < (float) (b[2] - b[0]) / (b[3] - b[1]); });
        for (size_t i = 0; i < bs.size(); i += 6) {
            size_t n = std::min<size_t>(6, bs.size() - i); float mr = 320.f / 48;
            for (size_t j = 0; j < n; j++) mr = std::max(mr, (float) (bs[i + j][2] - bs[i + j][0]) / (bs[i + j][3] - bs[i + j][1]));
            int RW = (int) (48 * mr);
            std::vector<float> r((size_t) n * 3 * 48 * RW, 0.f);
            for (size_t j = 0; j < n; j++) { auto &b = bs[i + j]; int w = b[2] - b[0], h = b[3] - b[1];
                resize_norm(img, iw, b[0], b[1], w, h, r.data() + j * 3 * 48 * RW, std::min(RW, (int) std::ceil(48.f * w / h)), 48, RW, rm, rs); }
            std::array<int64_t, 4> rsh{(int64_t) n, 3, 48, RW};
            Ort::Value rt = Ort::Value::CreateTensor<float>(mem, r.data(), r.size(), rsh.data(), 4);
            const char *ri[] = {rin.c_str()}, *ro[] = {rout.c_str()};
            auto rr = rec->Run(Ort::RunOptions{nullptr}, ri, &rt, 1, ro, 1);
            auto s = rr[0].GetTensorTypeAndShapeInfo().GetShape(); const float *o = rr[0].GetTensorData<float>();
            volatile long chk = 0;
            for (int64_t a = 0; a < s[0] * s[1]; a++) chk += std::max_element(o + a * s[2], o + (a + 1) * s[2]) - (o + a * s[2]);
        }
        stbi_image_free(img);
        return ms_since(t0);
    }
};

int main(int argc, char **argv) {
    if (argc < 11) { fprintf(stderr, "usage: liveloop DET REC PLAN FRAMES CROPS RULE PREFIX SUFFIX PORT THREADS [CADENCE_MS] [BUDGET]\n"); return 2; }
    std::string rule = argv[6], prefix = slurp(argv[7]), suffix = slurp(argv[8]);
    int port = atoi(argv[9]), threads = atoi(argv[10]);
    double cadence = argc > 11 ? atof(argv[11]) : 750.0; int budget = argc > 12 ? atoi(argv[12]) : 7;
    std::vector<Clip> clips;
    { std::ifstream in(argv[3]); std::string line;
      while (std::getline(in, line)) { std::istringstream ss(line); std::string tag; ss >> tag;
          if (tag == "C") { Clip c; int n; ss >> c.id >> n; clips.push_back(c); }
          else if (tag == "F") { Frame f; int nl; ss >> f.name >> f.level >> f.ocr >> nl; clips.back().frames.push_back(f); }
          else if (tag == "L") { std::array<int, 4> b; ss >> b[0] >> b[1] >> b[2] >> b[3]; clips.back().frames.back().boxes.push_back(b); } } }
    OCR ocr(argv[1], argv[2], threads);
    ocr.run(std::string(argv[4]) + "/" + clips[0].frames[0].name, clips[0].frames[0].boxes);   // warm-up
    struct VR { std::string date; double dur; clk::time_point done; };
    auto vlm = [&](const std::string &crop) {
        auto t0 = clk::now();
        std::string resp = post(port, prefix + b64(slurp(crop)) + suffix);
        return VR{parse_vlm(resp), ms_since(t0), clk::now()};
    };
    printf("clip\trule\tanswer\tdecision_ms\texamined\tvlm_calls\tlast_frame_idx\n");
    for (auto &c : clips) {
        int n = c.frames.size(), last = -1, examined = 0, calls = 0;
        std::map<std::string, int> seen; std::string answer = "ABSTAIN";
        std::future<VR> pend; int pend_idx = -1;
        auto t0 = clk::now(); double decided = -1;
        auto settle = [&](int idx, const std::string &v) {   // apply a finished VLM read of frame idx
            const Frame &f = c.frames[idx];
            if (rule == "xcheck") { if (v != "NONE" && v == f.ocr) return v; }
            else if (rule == "gated") { if (valid(v) && ++seen[v] == 2) return v; }
            return std::string();
        };
        auto finish_pending = [&]() {
            if (!pend.valid()) return false;
            auto r = pend.get(); calls++; std::string a = settle(pend_idx, r.date);
            double at = std::chrono::duration<double, std::milli>(r.done - t0).count();   // the read's own completion
            printf("#\t%s\t%s\t%d\t%.0f\tvlm\t%.0f\t%s\n", c.id.c_str(), rule.c_str(), pend_idx, at, r.dur, r.date.c_str());
            if (!a.empty()) { answer = a; decided = at; return true; }
            return false;
        };
        while (true) {
            if (examined >= budget) { finish_pending(); break; }           // budget spent: only the last read can still decide
            if (pend.valid() && pend.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
                if (finish_pending()) break;                              // a read finished before the next frame
            double now = ms_since(t0);
            int arrived = std::min(n - 1, (int) std::floor(now / cadence));
            if (arrived <= last) {                                        // nothing new has arrived
                if (last == n - 1) { finish_pending(); break; }          // clip over
                std::this_thread::sleep_for(std::chrono::milliseconds(5)); continue;
            }
            int j = arrived; last = j; examined++;                         // latest frame; older ones are dropped
            const Frame &f = c.frames[j];
            double tocr = ocr.run(std::string(argv[4]) + "/" + f.name, f.boxes);
            printf("#\t%s\t%s\t%d\t%.0f\tocr\t%.0f\n", c.id.c_str(), rule.c_str(), j, ms_since(t0), tocr);
            if (finish_pending()) break;                                  // the earlier frame's read decides first
            if (rule == "ocr2") {
                if (f.ocr != "NONE" && ++seen[f.ocr] == 2) { answer = f.ocr; decided = ms_since(t0); break; }
            } else {
                bool need = rule == "xcheck" ? f.ocr != "NONE" : (f.level == "date" || f.level == "digits");
                if (need) { std::string crop = std::string(argv[5]) + "/" + f.name;
                    pend = std::async(std::launch::async, vlm, crop); pend_idx = j; }
            }
        }
        if (decided < 0) decided = ms_since(t0);
        printf("%s\t%s\t%s\t%.0f\t%d\t%d\t%d\n", c.id.c_str(), rule.c_str(), answer.c_str(), decided, examined, calls, last);
        fflush(stdout);
        if (pend.valid()) pend.wait();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return 0;
}
