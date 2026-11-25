// Live decision loop on the headset with on-device localisation (onloc.hpp). For each recorded clip, sampled frames
// "arrive" at their recorded cadence (0.75 s apart); the loop always takes the latest frame that has arrived and drops
// older ones, decodes it, runs the full OCR pipeline on the CPU (detector, line boxes, recogniser, decoding, gate level,
// date rules, crop), and, when the rule needs a read, encodes the 2x crop and sends it to the persistent llama-server
// (GPU) asynchronously, so localisation of the next frame overlaps with the read. Nothing comes from the offline
// pipeline: the plan file only lists each clip's frames.
//
//   liveloop2 DET REC DICT PLAN FRAME_DIR RULE PREFIX SUFFIX PORT THREADS [CADENCE_MS=750] [BUDGET=7]
// RULE: xcheck   VLM read only on frames with an OCR date; answer when the VLM's date equals the OCR date
//       gated    VLM read only on frames with a date-like or digit line; answer when a valid date (or "no date") is read twice
//       ocronly  no VLM; decides three OCR-only rules on the same frames: ocr2 (first OCR date seen on 2 frames),
//                conf2_99 (2 frames, line confidence >= 0.99) and conf3_80 (3 frames, confidence >= 0.80)
// PLAN: "C clip n" then n lines "F frame ..." (other fields ignored). Output: one TSV row per clip and rule, and one
// '#' row per examined frame and per finished read.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "onloc.hpp"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <future>
#include <sstream>
#include <thread>

using onloc::clk;
static double ms_since(clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); }
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

static std::string post(int port, const std::string &body) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port); inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    if (connect(s, (sockaddr *) &a, sizeof a) < 0) { close(s); return ""; }
    std::string req = "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
                      + std::to_string(body.size()) + "\r\n\r\n" + body;
    for (size_t o = 0; o < req.size();) { ssize_t n = send(s, req.data() + o, req.size() - o, 0); if (n <= 0) break; o += n; }
    std::string resp; char buf[8192]; ssize_t n;
    while ((n = recv(s, buf, sizeof buf, 0)) > 0) resp.append(buf, n);
    close(s); return resp;
}
static std::string parse_vlm(const std::string &resp) {
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

struct Clip { std::string id; std::vector<std::string> frames; };
struct OcrRule { const char *name; int k; float tau; };

int main(int argc, char **argv) {
    if (argc < 11) { fprintf(stderr, "usage: liveloop2 DET REC DICT PLAN FRAMES RULE PREFIX SUFFIX PORT THREADS [CADENCE_MS] [BUDGET]\n"); return 2; }
    std::string rule = argv[6], prefix = slurp(argv[7]), suffix = slurp(argv[8]), fdir = argv[5];
    int port = atoi(argv[9]), threads = atoi(argv[10]);
    double cadence = argc > 11 ? atof(argv[11]) : 750.0; int budget = argc > 12 ? atoi(argv[12]) : 7;
    std::vector<Clip> clips;
    { std::ifstream in(argv[4]); std::string line;
      while (std::getline(in, line)) { std::istringstream ss(line); std::string tag; ss >> tag;
          if (tag == "C") { Clip c; int n; ss >> c.id >> n; clips.push_back(c); }
          else if (tag == "F") { std::string f; ss >> f; clips.back().frames.push_back(f); } } }
    onloc::Pipeline pl(argv[1], argv[2], argv[3], threads);
    { int iw, ih, ch; unsigned char *img = stbi_load((fdir + "/" + clips[0].frames[0]).c_str(), &iw, &ih, &ch, 3); pl.run(img, iw, ih, false); stbi_image_free(img); }   // warm-up
    const std::vector<OcrRule> ocr_rules = {{"ocr2", 2, 0.f}, {"conf2_99", 2, 0.99f}, {"conf3_80", 3, 0.80f}};
    struct VR { std::string date; double dur; clk::time_point done; };
    auto vlm = [&](std::string jpeg) {
        auto t0 = clk::now();
        std::string resp = post(port, prefix + b64(jpeg) + suffix);
        return VR{parse_vlm(resp), ms_since(t0), clk::now()};
    };
    printf("clip\trule\tanswer\tdecision_ms\texamined\tvlm_calls\tlast_frame_idx\n");
    for (auto &c : clips) {
        int n = c.frames.size(), last = -1, examined = 0, calls = 0;
        std::vector<std::string> fdate(n), flevel(n);
        std::map<std::string, int> seen; std::string answer = "ABSTAIN";
        std::vector<std::map<std::string, int>> oseen(ocr_rules.size());
        std::vector<std::string> oans(ocr_rules.size(), "ABSTAIN"); std::vector<double> odec(ocr_rules.size(), -1); std::vector<int> oex(ocr_rules.size(), 0);
        std::future<VR> pend; int pend_idx = -1;
        auto t0 = clk::now(); double decided = -1;
        auto settle = [&](int idx, const std::string &v) {
            if (rule == "xcheck") { if (v != "NONE" && v == fdate[idx]) return v; }
            else if (rule == "gated") { if (valid(v) && ++seen[v] == 2) return v; }
            return std::string();
        };
        auto finish_pending = [&]() {
            if (!pend.valid()) return false;
            auto r = pend.get(); calls++; std::string a = settle(pend_idx, r.date);
            double at = std::chrono::duration<double, std::milli>(r.done - t0).count();
            printf("#\t%s\t%s\t%d\t%.0f\tvlm\t%.0f\t%s\n", c.id.c_str(), rule.c_str(), pend_idx, at, r.dur, r.date.c_str());
            if (!a.empty()) { answer = a; decided = at; return true; }
            return false;
        };
        auto all_ocr_decided = [&]() { for (auto d : odec) if (d < 0) return false; return true; };
        while (true) {
            if (examined >= budget) { finish_pending(); break; }
            if (pend.valid() && pend.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
                if (finish_pending()) break;
            double now = ms_since(t0);
            int arrived = std::min(n - 1, (int) std::floor(now / cadence));
            if (arrived <= last) {
                if (last == n - 1) { finish_pending(); break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(5)); continue;
            }
            int j = arrived; last = j; examined++;
            auto ts = clk::now();
            int iw, ih, ch; unsigned char *img = stbi_load((fdir + "/" + c.frames[j]).c_str(), &iw, &ih, &ch, 3);
            if (!img) { fprintf(stderr, "cannot read %s\n", c.frames[j].c_str()); continue; }
            auto R = pl.run(img, iw, ih, false);
            fdate[j] = R.date; flevel[j] = R.level;
            bool need = rule == "xcheck" ? R.date != "NONE" : rule == "gated" ? (R.level == "date" || R.level == "digits") : false;
            std::string jpeg; if (need) jpeg = onloc::crop_x2_jpeg(img, iw, R.crop);
            stbi_image_free(img);
            double tocr = ms_since(ts);
            printf("#\t%s\t%s\t%d\t%.0f\tocr\t%.0f\t%s\t%s\t%.3f\t%zu\t%.0f\t%.0f\t%.0f\t%d,%d,%d,%d\n", c.id.c_str(), rule.c_str(), j, ms_since(t0), tocr,
                   R.level.c_str(), R.date.c_str(), R.conf, R.lines.size(), R.t_det, R.t_rec, R.t_post, R.crop[0], R.crop[1], R.crop[2], R.crop[3]);
            if (finish_pending()) break;                                  // the earlier frame's read decides first
            if (rule == "ocronly") {
                for (size_t k = 0; k < ocr_rules.size(); k++)
                    if (odec[k] < 0 && R.date != "NONE" && R.conf >= ocr_rules[k].tau && ++oseen[k][R.date] == ocr_rules[k].k) { oans[k] = R.date; odec[k] = ms_since(t0); oex[k] = examined; }
                if (all_ocr_decided()) break;
            } else if (need) { pend = std::async(std::launch::async, vlm, std::move(jpeg)); pend_idx = j; }
        }
        double end = ms_since(t0);
        if (rule == "ocronly") {
            for (size_t k = 0; k < ocr_rules.size(); k++)
                printf("%s\t%s\t%s\t%.0f\t%d\t%d\t%d\n", c.id.c_str(), ocr_rules[k].name, oans[k].c_str(), odec[k] < 0 ? end : odec[k], odec[k] < 0 ? examined : oex[k], 0, last);
        } else {
            if (decided < 0) decided = end;
            printf("%s\t%s\t%s\t%.0f\t%d\t%d\t%d\n", c.id.c_str(), rule.c_str(), answer.c_str(), decided, examined, calls, last);
        }
        fflush(stdout);
        if (pend.valid()) pend.wait();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return 0;
}
