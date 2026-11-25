// On-device localisation, OCR date and VLM crop: a C++ port of the offline pipeline used for the real clips
// (PaddleOCR 3 "OCR" pipeline with PP-OCRv5_mobile_det + latin_PP-OCRv5_mobile_rec, no orientation models; then
// loc_clip.locate() for the gate level and the crop, and ocr_baseline.select() for the OCR date).
//   detector:  DB post-processing (thresh 0.3, box_thresh 0.6, unclip 1.5, min size 3), minimum-area rectangles of
//              8-connected components, "fast" box score, quad ordering and top-to-bottom/left-to-right sort.
//   recogniser: rectified line crop (rotated 90 deg if tall), height 48, batches of 6 sorted by aspect ratio and
//              zero-padded to the batch's widest, CTC greedy decoding with the model's dictionary.
//   locate:    same regular expressions and geometry as loc_clip.py; crop of the frame, 2x bicubic, JPEG q95.
//   date:      same patterns, keywords and scoring as ocr_baseline.py.
// Requires stb_image.h (STB_IMAGE_IMPLEMENTATION defined by the including file) and stb_image_write.h.
#pragma once
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <vector>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "normtab.hpp"
#include <chrono>

namespace onloc {

struct P { double x, y; };
struct Line { std::string text; float score; std::array<int, 4> box; std::array<P, 4> quad; };
struct Result {
    std::vector<Line> lines; std::string level, date; float conf = 0; std::array<int, 4> crop; std::string jpeg;   // x2 crop as JPEG
    double t_det = 0, t_rec = 0, t_post = 0;
};

// ---------- geometry ----------
static double cross(const P &o, const P &a, const P &b) { return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x); }
static std::vector<P> hull(std::vector<P> p) {
    std::sort(p.begin(), p.end(), [](const P &a, const P &b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });
    p.erase(std::unique(p.begin(), p.end(), [](const P &a, const P &b) { return a.x == b.x && a.y == b.y; }), p.end());
    if (p.size() < 3) return p;
    std::vector<P> h(2 * p.size()); size_t k = 0;
    for (size_t i = 0; i < p.size(); i++) { while (k >= 2 && cross(h[k - 2], h[k - 1], p[i]) <= 0) k--; h[k++] = p[i]; }
    for (size_t i = p.size() - 1, t = k + 1; i > 0; i--) { while (k >= t && cross(h[k - 2], h[k - 1], p[i - 1]) <= 0) k--; h[k++] = p[i - 1]; }
    h.resize(k - 1); return h;
}
struct RRect { P c; double w, h; double ux, uy; };   // centre, side lengths along u=(ux,uy) and its normal
static RRect min_area_rect(const std::vector<P> &pts) {
    std::vector<P> h = hull(pts);
    if (h.size() == 1) return {h[0], 0, 0, 1, 0};
    if (h.size() == 2) { double dx = h[1].x - h[0].x, dy = h[1].y - h[0].y, L = std::hypot(dx, dy);
        return {{(h[0].x + h[1].x) / 2, (h[0].y + h[1].y) / 2}, L, 0, dx / L, dy / L}; }
    RRect best{{0, 0}, 0, 0, 1, 0}; double ba = 1e300;
    for (size_t i = 0; i < h.size(); i++) {
        const P &a = h[i], &b = h[(i + 1) % h.size()];
        double dx = b.x - a.x, dy = b.y - a.y, L = std::hypot(dx, dy); if (L == 0) continue;
        double ux = dx / L, uy = dy / L, mn1 = 1e300, mx1 = -1e300, mn2 = 1e300, mx2 = -1e300;
        for (auto &q : h) { double s = q.x * ux + q.y * uy, t = -q.x * uy + q.y * ux;
            mn1 = std::min(mn1, s); mx1 = std::max(mx1, s); mn2 = std::min(mn2, t); mx2 = std::max(mx2, t); }
        double area = (mx1 - mn1) * (mx2 - mn2);
        if (area < ba) { ba = area; double s = (mn1 + mx1) / 2, t = (mn2 + mx2) / 2;
            best = {{s * ux - t * uy, s * uy + t * ux}, mx1 - mn1, mx2 - mn2, ux, uy}; }
    }
    return best;
}
static std::array<P, 4> rect_points(const RRect &r) {   // corners, then PaddleOCR's get_mini_boxes ordering
    double hx = r.w / 2, hy = r.h / 2; std::array<P, 4> p;
    const double s[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    for (int i = 0; i < 4; i++) p[i] = {r.c.x + s[i][0] * hx * r.ux - s[i][1] * hy * r.uy, r.c.y + s[i][0] * hx * r.uy + s[i][1] * hy * r.ux};
    std::sort(p.begin(), p.end(), [](const P &a, const P &b) { return a.x < b.x; });
    int i1, i2, i3, i4;
    if (p[1].y > p[0].y) { i1 = 0; i4 = 1; } else { i1 = 1; i4 = 0; }
    if (p[3].y > p[2].y) { i2 = 2; i3 = 3; } else { i2 = 3; i3 = 2; }
    return {p[i1], p[i2], p[i3], p[i4]};
}
static bool inside(const std::array<P, 4> &q, double x, double y) {   // convex quad, either orientation
    int pos = 0, neg = 0;
    for (int i = 0; i < 4; i++) { double c = cross(q[i], q[(i + 1) % 4], {x, y}); if (c > 1e-9) pos++; else if (c < -1e-9) neg++; }
    return !(pos && neg);
}

// ---------- text ----------
static std::vector<uint32_t> utf8_decode(const std::string &s) {
    std::vector<uint32_t> o;
    for (size_t i = 0; i < s.size();) { unsigned char c = s[i]; uint32_t cp; int n;
        if (c < 0x80) { cp = c; n = 1; } else if ((c >> 5) == 6) { cp = c & 31; n = 2; } else if ((c >> 4) == 14) { cp = c & 15; n = 3; } else { cp = c & 7; n = 4; }
        for (int k = 1; k < n && i + k < s.size(); k++) cp = (cp << 6) | (s[i + k] & 63);
        o.push_back(cp); i += n; }
    return o;
}
// NFKD + drop non-ASCII + upper (ocr_baseline.norm_text) for the Latin letters the recogniser can emit
static std::string norm_text(const std::string &s) {
    static std::map<uint32_t, std::string> base;
    if (base.empty()) for (auto &e : NORMTAB) base[e.first] = e.second;
    std::string o;
    for (uint32_t cp : utf8_decode(s)) {
        if (cp < 0x80) o += (char) cp;
        else { auto it = base.find(cp); if (it != base.end()) o += it->second; }
    }
    for (auto &c : o) c = (char) toupper((unsigned char) c);
    return o;
}

// ---------- ocr_baseline.select ----------
struct Date { int y, m, d; };   // d = 0: no day
static std::map<std::string, int> months() {
    std::map<std::string, int> M; const char *names[] = {"JAN ENE GEN JAN", "FEB FEV FEB", "MAR MAR MRZ MAE", "APR ABR AVR APR",
        "MAY MAI MAG MEI", "JUN JUIN GIU", "JUL JUIL LUG", "AUG AGO AOU AGO", "SEP SET SEPT", "OCT OUT OTT OKT", "NOV", "DEC DIC DEZ DES"};
    for (int i = 0; i < 12; i++) { std::string s = names[i]; size_t a = 0;
        while (a < s.size()) { size_t b = s.find(' ', a); if (b == std::string::npos) b = s.size(); M[s.substr(a, b - a)] = i + 1; a = b + 1; } }
    return M;
}
static int to_year(const std::string &y) { int v = std::stoi(y); return v < 100 ? v + 2000 : v; }
static bool valid_date(int y, int m, int d) { return 2020 <= y && y <= 2040 && 1 <= m && m <= 12 && (d == 0 || (1 <= d && d <= 31)); }
static std::vector<Date> parse_dates(const std::string &text, bool compact) {
    static const std::map<std::string, int> M = months();
    static const std::string SEP = "[\\s./\\-]{1,2}", Y4 = "(20\\d{2})", Y2 = "(\\d{2})", NUM = "(\\d{1,2})", MON = "([A-Z]{3,5})\\.?";
    static const std::vector<std::pair<std::regex, std::string>> PAT = {
        {std::regex("\\b" + Y4 + SEP + NUM + SEP + NUM + "\\b"), "ymd"},
        {std::regex("\\b" + NUM + SEP + NUM + SEP + Y4 + "\\b"), "dmy"},
        {std::regex("\\b" + NUM + SEP + NUM + SEP + Y2 + "\\b"), "dmy"},
        {std::regex("\\b" + NUM + SEP + "?" + MON + SEP + "?" + Y4 + "\\b"), "dMy"},
        {std::regex("\\b" + NUM + SEP + "?" + MON + SEP + "?" + Y2 + "\\b"), "dMy"},
        {std::regex("\\b" + MON + SEP + "?" + Y4 + "\\b"), "My"},
        {std::regex("\\b" + Y4 + SEP + NUM + "\\b"), "ym"},
        {std::regex("\\b" + NUM + SEP + Y4 + "\\b"), "my"},
        {std::regex("\\b" + NUM + "[./\\-]" + Y2 + "\\b"), "my2"}};
    static const std::pair<std::regex, std::string> COMPACT = {std::regex("\\b(\\d{2})(\\d{2})(20\\d{2}|\\d{2})\\b"), "dmy"};
    std::string t = norm_text(text);
    std::vector<Date> found; std::vector<std::pair<long, long>> taken;
    auto pats = PAT; if (compact) pats.push_back(COMPACT);
    for (auto &[rx, order] : pats) {
        for (auto it = std::sregex_iterator(t.begin(), t.end(), rx); it != std::sregex_iterator(); ++it) {
            auto &mt = *it; long s = mt.position(0), e = s + mt.length(0);
            bool ov = false; for (auto &[a, b] : taken) if (a < e && s < b) ov = true;
            if (ov) continue;
            int y, m, d;
            if (order == "ymd") { y = to_year(mt[1]); m = std::stoi(mt[2]); d = std::stoi(mt[3]); }
            else if (order == "dmy") { d = std::stoi(mt[1]); m = std::stoi(mt[2]); y = to_year(mt[3]); }
            else if (order == "dMy") { auto k = M.find(mt[2].str().substr(0, 3)); if (k == M.end()) continue; d = std::stoi(mt[1]); m = k->second; y = to_year(mt[3]); }
            else if (order == "My") { auto k = M.find(mt[1].str().substr(0, 3)); if (k == M.end()) continue; d = 0; m = k->second; y = to_year(mt[2]); }
            else if (order == "ym") { d = 0; m = std::stoi(mt[2]); y = to_year(mt[1]); }
            else if (order == "my") { d = 0; m = std::stoi(mt[1]); y = to_year(mt[2]); }
            else { int yy = std::stoi(mt[2]); if (yy < 24 || yy > 35) continue; d = 0; m = std::stoi(mt[1]); y = to_year(mt[2]); }
            if (valid_date(y, m, d)) { found.push_back({y, m, d}); taken.push_back({s, e}); }
        }
    }
    return found;
}
static bool has_kw(const std::string &text, const std::vector<std::string> &kws) {
    std::string t = " ";
    for (char c : norm_text(text)) t += ((c >= 'A' && c <= 'Z') || c == '.' || c == ' ') ? c : ' ';
    t += " ";
    for (auto &k : kws) for (size_t p = t.find(k); p != std::string::npos; p = t.find(k, p + 1)) {
        bool lb = p == 0 || !(t[p - 1] >= 'A' && t[p - 1] <= 'Z'), rb = p + k.size() >= t.size() || !(t[p + k.size()] >= 'A' && t[p + k.size()] <= 'Z');
        if (lb && rb) return true; }
    return false;
}
static std::string ocr_date(const std::vector<Line> &lines, float *conf = nullptr) {
    static const std::vector<std::string> EXPIRY = {"CAD", "CADUC", "CONSUMIR", "CONS PREF", "CONS.PREF", "PREFERENTEMENTE", "PREFERENCIA",
        "ANTES", "BEST BEFORE", "BEST BY", "BB", "BBE", "EXP", "USE BY", "VAL", "VENC", "VTO", "DLC", "DLUO", "DDM", "CONSOMMER", "CONSUMARSI",
        "SCAD", "MHD", "MINDESTENS", "HALTBAR", "FC", "F.C", "C.P", "CP"};
    static const std::vector<std::string> PRODUCTION = {"ENV", "ENVASADO", "FAB", "FABRIC", "ELAB", "PROD", "LOTE", "LOT", "PACKED", "PKD", "MFG", "MFD", "EMB"};
    bool have = false; int bs = 0; Date bd{}; float bc = 0;
    for (size_t i = 0; i < lines.size(); i++) {
        bool near = has_kw(lines[i].text, EXPIRY) || (i > 0 && has_kw(lines[i - 1].text, EXPIRY));
        for (auto &dt : parse_dates(lines[i].text, near)) {
            int sc = (near ? 2 : 0) - (has_kw(lines[i].text, PRODUCTION) ? 2 : 0);
            auto key = std::make_tuple(sc, dt.y, dt.m, dt.d), bk = std::make_tuple(bs, bd.y, bd.m, bd.d);
            if (!have || key > bk) { have = true; bs = sc; bd = dt; bc = lines[i].score; }   // first maximum wins, as Python's max()
        }
    }
    if (conf) *conf = have ? bc : 0.f;
    if (!have) return "NONE";
    char b[16]; if (bd.d) snprintf(b, sizeof b, "%04d-%02d-%02d", bd.y, bd.m, bd.d); else snprintf(b, sizeof b, "%04d-%02d-??", bd.y, bd.m);
    return b;
}

// ---------- loc_clip.locate ----------
static int date_score(const std::string &t) {
    static const std::string MONTH = "(jan|ene|feb|fev|mar|abr|apr|avr|may|mai|jun|jui|jul|ago|aug|aou|sep|set|oct|out|okt|nov|dic|dec|dez)";
    static const std::vector<std::pair<int, std::regex>> R = {
        {3, std::regex("\\b\\d{1,2}\\s*[./\\-]\\s*\\d{1,2}\\s*[./\\-]\\s*\\d{2,4}\\b")},
        {3, std::regex("\\d{1,2}\\s*" + MONTH + "\\w*\\.?\\s*\\d{2,4}|" + MONTH + "\\w*\\.?\\s*\\d{2,4}", std::regex::icase)},
        {2, std::regex("\\b\\d{1,2}\\s*[./\\-]\\s*20[2-4]\\d\\b")},
        {1, std::regex("\\b\\d{1,2}\\s*[/\\-]\\s*\\d{2}\\b")},
        {1, std::regex("\\b20[2-4]\\d\\b")}};
    int best = 0; for (auto &[sc, rx] : R) if (sc > best && std::regex_search(t, rx)) best = sc;
    return best;
}
static bool is_key(const std::string &t) {
    static const std::regex K("cad|consum|exp|best|use\\s*by|\\bbb|\\bbbe|venc|val|prefer|mhd|dlc|dluo|ddm|fecha|date", std::regex::icase);
    return std::regex_search(t, K);
}
static void locate(const std::vector<Line> &lines, int W, int H, std::array<int, 4> &crop, std::string &level) {
    std::vector<int> sc(lines.size()); bool any = false; level = "date";
    for (size_t i = 0; i < lines.size(); i++) { sc[i] = date_score(lines[i].text); any |= sc[i] > 0; }
    if (!any) { level = "digits";
        for (size_t i = 0; i < lines.size(); i++) { int nd = 0; for (char c : lines[i].text) nd += c >= '0' && c <= '9'; sc[i] = nd >= 4; any |= sc[i] > 0; } }
    if (!any) { crop = {0, 0, W, H}; level = "fallback_whole"; return; }
    std::vector<size_t> keys; for (size_t i = 0; i < lines.size(); i++) if (is_key(lines[i].text)) keys.push_back(i);
    auto cx = [&](const std::array<int, 4> &b) { return (b[0] + b[2]) / 2.0; };
    auto cy = [&](const std::array<int, 4> &b) { return (b[1] + b[3]) / 2.0; };
    auto near = [&](const std::array<int, 4> &b, const std::array<int, 4> &r, double reach) { return std::abs(cx(b) - cx(r)) <= reach && std::abs(cy(b) - cy(r)) <= reach; };
    auto reach_of = [&](const std::array<int, 4> &b) { return 2.0 * std::max({b[2] - b[0], b[3] - b[1], 32}); };
    int bi = -1; double bt = -1e9;
    for (size_t i = 0; i < lines.size(); i++) {
        double tot = -1;
        if (sc[i]) { tot = sc[i]; double r = reach_of(lines[i].box); for (size_t k : keys) if (near(lines[k].box, lines[i].box, r)) { tot += 2; break; } }
        if (tot > bt) { bt = tot; bi = (int) i; }
    }
    const auto &ref = lines[bi].box; double reach = reach_of(ref);
    std::vector<std::array<int, 4>> ch;
    for (size_t i = 0; i < lines.size(); i++) if (sc[i] && near(lines[i].box, ref, reach)) ch.push_back(lines[i].box);
    for (size_t k : keys) if (near(lines[k].box, ref, reach)) ch.push_back(lines[k].box);
    double b[4] = {1e9, 1e9, -1e9, -1e9};
    for (auto &x : ch) { b[0] = std::min(b[0], (double) x[0]); b[1] = std::min(b[1], (double) x[1]); b[2] = std::max(b[2], (double) x[2]); b[3] = std::max(b[3], (double) x[3]); }
    double pad = 0.25 * std::min(b[2] - b[0], b[3] - b[1]) + 16;
    b[0] -= pad; b[1] -= pad; b[2] += pad; b[3] += pad;
    for (int a = 0; a < 2; a++) if (b[a + 2] - b[a] < 96) { double c = (b[a] + b[a + 2]) / 2; b[a] = c - 48; b[a + 2] = c + 48; }
    crop = {std::max(0, (int) b[0]), std::max(0, (int) b[1]), std::min(W, (int) b[2]), std::min(H, (int) b[3])};
}

// ---------- 2x bicubic (PIL BICUBIC, a = -0.5) and JPEG ----------
static double cubic(double x) { const double a = -0.5; x = std::abs(x);
    if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1; if (x < 2) return (((x - 5) * x + 8) * x - 4) * a; return 0; }
static std::string crop_x2_jpeg(const unsigned char *img, int iw, const std::array<int, 4> &c) {
    int w = c[2] - c[0], h = c[3] - c[1], W = 2 * w, H = 2 * h;
    std::vector<float> tmp((size_t) h * W * 3); std::vector<unsigned char> out((size_t) H * W * 3);
    auto weights = [&](int o, int n, int &lo, double ww[4]) { double s = (o + 0.5) / 2.0 - 0.5; lo = (int) std::floor(s) - 1; double sum = 0;
        for (int k = 0; k < 4; k++) { ww[k] = cubic(s - (lo + k)); sum += ww[k]; } for (int k = 0; k < 4; k++) ww[k] /= sum; (void) n; };
    for (int x = 0; x < W; x++) { int lo; double ww[4]; weights(x, w, lo, ww);
        for (int y = 0; y < h; y++) for (int ch = 0; ch < 3; ch++) { double v = 0;
            for (int k = 0; k < 4; k++) { int xx = std::clamp(lo + k, 0, w - 1); v += ww[k] * img[((size_t) (c[1] + y) * iw + c[0] + xx) * 3 + ch]; }
            tmp[((size_t) y * W + x) * 3 + ch] = (float) v; } }
    for (int y = 0; y < H; y++) { int lo; double ww[4]; weights(y, h, lo, ww);
        for (int x = 0; x < W; x++) for (int ch = 0; ch < 3; ch++) { double v = 0;
            for (int k = 0; k < 4; k++) { int yy = std::clamp(lo + k, 0, h - 1); v += ww[k] * tmp[((size_t) yy * W + x) * 3 + ch]; }
            out[((size_t) y * W + x) * 3 + ch] = (unsigned char) std::clamp((int) std::lround(v), 0, 255); } }
    std::string jpg;
    stbi_write_jpg_to_func([](void *ctx, void *data, int size) { ((std::string *) ctx)->append((const char *) data, size); }, &jpg, W, H, 3, out.data(), 95);
    return jpg;
}


// homography mapping (0,0),(w,0),(w,h),(0,h) to q[0..3] (cv2.getPerspectiveTransform(dst->src))
static std::array<double, 9> homography(const std::array<P, 4> &q, double w, double h) {
    double A[8][9] = {}; const double X[4] = {0, w, w, 0}, Y[4] = {0, 0, h, h};
    for (int i = 0; i < 4; i++) {
        double *r1 = A[2 * i], *r2 = A[2 * i + 1];
        r1[0] = X[i]; r1[1] = Y[i]; r1[2] = 1; r1[6] = -X[i] * q[i].x; r1[7] = -Y[i] * q[i].x; r1[8] = q[i].x;
        r2[3] = X[i]; r2[4] = Y[i]; r2[5] = 1; r2[6] = -X[i] * q[i].y; r2[7] = -Y[i] * q[i].y; r2[8] = q[i].y;
    }
    for (int c = 0; c < 8; c++) { int pv = c; for (int r = c + 1; r < 8; r++) if (std::abs(A[r][c]) > std::abs(A[pv][c])) pv = r;
        for (int k = 0; k < 9; k++) std::swap(A[c][k], A[pv][k]);
        for (int r = 0; r < 8; r++) if (r != c && A[c][c] != 0) { double f = A[r][c] / A[c][c]; for (int k = c; k < 9; k++) A[r][k] -= f * A[c][k]; } }
    std::array<double, 9> Hm; for (int i = 0; i < 8; i++) Hm[i] = A[i][8] / A[i][i]; Hm[8] = 1; return Hm;
}
static double cubw(double x) { const double a = -0.75; x = std::abs(x);
    if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1; if (x < 2) return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; return 0; }
// cv2.warpPerspective(..., INTER_CUBIC, BORDER_REPLICATE): uint8 RGB crop of size cw x ch
static std::vector<unsigned char> warp_cubic(const unsigned char *img, int iw, int ih, const std::array<P, 4> &q, int cw, int ch) {
    auto Hm = homography(q, cw, ch); std::vector<unsigned char> out((size_t) cw * ch * 3);
    for (int y = 0; y < ch; y++) for (int x = 0; x < cw; x++) {
        double W = Hm[6] * x + Hm[7] * y + Hm[8], sx = (Hm[0] * x + Hm[1] * y + Hm[2]) / W, sy = (Hm[3] * x + Hm[4] * y + Hm[5]) / W;
        sx = std::round(sx * 32) / 32; sy = std::round(sy * 32) / 32;   // INTER_BITS = 5
        int x0 = (int) std::floor(sx), y0 = (int) std::floor(sy); double fx = sx - x0, fy = sy - y0, wx[4], wy[4];
        for (int k = 0; k < 4; k++) { wx[k] = cubw(fx - (k - 1)); wy[k] = cubw(fy - (k - 1)); }
        for (int c = 0; c < 3; c++) { double v = 0;
            for (int j = 0; j < 4; j++) { int yy = std::clamp(y0 - 1 + j, 0, ih - 1);
                for (int i = 0; i < 4; i++) { int xx = std::clamp(x0 - 1 + i, 0, iw - 1); v += wy[j] * wx[i] * img[((size_t) yy * iw + xx) * 3 + c]; } }
            out[((size_t) y * cw + x) * 3 + c] = (unsigned char) std::clamp((int) std::lround(v), 0, 255); }
    }
    return out;
}
static std::vector<unsigned char> rot90ccw(const std::vector<unsigned char> &a, int w, int h) {   // np.rot90: (h,w) -> (w,h)
    std::vector<unsigned char> o(a.size());
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) for (int c = 0; c < 3; c++) o[((size_t) (w - 1 - x) * h + y) * 3 + c] = a[((size_t) y * w + x) * 3 + c];
    return o;
}
// cv2.resize INTER_LINEAR to (ow, oh), uint8
static std::vector<unsigned char> resize_linear(const std::vector<unsigned char> &a, int w, int h, int ow, int oh) {
    std::vector<unsigned char> o((size_t) ow * oh * 3); double sx = (double) w / ow, sy = (double) h / oh;
    for (int y = 0; y < oh; y++) { double fy = (y + 0.5) * sy - 0.5; int y0 = (int) std::floor(fy); fy -= y0; if (y0 < 0) { y0 = 0; fy = 0; } if (y0 >= h - 1) { y0 = h - 1; fy = 0; }
        int y1 = std::min(y0 + 1, h - 1);
        for (int x = 0; x < ow; x++) { double fx = (x + 0.5) * sx - 0.5; int x0 = (int) std::floor(fx); fx -= x0; if (x0 < 0) { x0 = 0; fx = 0; } if (x0 >= w - 1) { x0 = w - 1; fx = 0; }
            int x1 = std::min(x0 + 1, w - 1);
            for (int c = 0; c < 3; c++) { auto A = [&](int yy, int xx) { return (double) a[((size_t) yy * w + xx) * 3 + c]; };
                double v = (1 - fy) * ((1 - fx) * A(y0, x0) + fx * A(y0, x1)) + fy * ((1 - fx) * A(y1, x0) + fx * A(y1, x1));
                o[((size_t) y * ow + x) * 3 + c] = (unsigned char) std::clamp((int) std::lround(v), 0, 255); } } }
    return o;
}

// ---------- the pipeline ----------
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

struct Pipeline {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "onloc"};
    Ort::Session *det, *rec;
    Ort::AllocatorWithDefaultOptions alloc;
    std::string din, dout, rin, rout;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<std::string> dict;   // index 0 = blank, then the dictionary, then " "
    Pipeline(const char *d, const char *r, const char *dict_path, int threads) {
        Ort::SessionOptions so; so.SetIntraOpNumThreads(threads); so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        det = new Ort::Session(env, d, so); rec = new Ort::Session(env, r, so);
        din = det->GetInputNameAllocated(0, alloc).get(); dout = det->GetOutputNameAllocated(0, alloc).get();
        rin = rec->GetInputNameAllocated(0, alloc).get(); rout = rec->GetOutputNameAllocated(0, alloc).get();
        dict.push_back(""); std::ifstream f(dict_path); std::string l; while (std::getline(f, l)) dict.push_back(l); dict.push_back(" ");
    }
    // bilinear sample (cv2-style, replicate border), channel order of the source (RGB)
    static float samp(const unsigned char *img, int iw, int ih, double x, double y, int ch) {
        int x0 = (int) std::floor(x), y0 = (int) std::floor(y); double fx = x - x0, fy = y - y0;
        auto Px = [&](int xx, int yy) { xx = std::clamp(xx, 0, iw - 1); yy = std::clamp(yy, 0, ih - 1); return (double) img[((size_t) yy * iw + xx) * 3 + ch]; };
        return (float) ((1 - fy) * ((1 - fx) * Px(x0, y0) + fx * Px(x0 + 1, y0)) + fy * ((1 - fx) * Px(x0, y0 + 1) + fx * Px(x0 + 1, y0 + 1)));
    }
    Result run(const unsigned char *img, int iw, int ih, bool want_crop = true) {
        Result R; auto t0 = clk::now();
        // detector input: size rounded to a multiple of 32 (limit_side_len 64, type min: no other resize at 1728 px)
        int W = std::max(32, (int) std::round(iw / 32.0) * 32), H = std::max(32, (int) std::round(ih / 32.0) * 32);
        const float dm[3] = {0.485f, 0.456f, 0.406f}, ds[3] = {0.229f, 0.224f, 0.225f};
        std::vector<float> x((size_t) 3 * H * W);
        if (W == iw && H == ih) {
            for (int c = 0; c < 3; c++) { float *d = x.data() + (size_t) c * H * W; const float sc = 1.f / (255.f * ds[c]), off = dm[c] / ds[c];
                for (size_t i = 0; i < (size_t) H * W; i++) d[i] = img[i * 3 + (2 - c)] * sc - off; }
        } else
        for (int yy = 0; yy < H; yy++) for (int xx = 0; xx < W; xx++) for (int c = 0; c < 3; c++) {
            double sx = (W == iw) ? xx : (xx + 0.5) * iw / W - 0.5, sy = (H == ih) ? yy : (yy + 0.5) * ih / H - 0.5;
            x[(size_t) c * H * W + (size_t) yy * W + xx] = (samp(img, iw, ih, sx, sy, 2 - c) / 255.f - dm[c]) / ds[c]; }   // BGR
        std::array<int64_t, 4> sh{1, 3, H, W};
        Ort::Value t = Ort::Value::CreateTensor<float>(mem, x.data(), x.size(), sh.data(), 4);
        const char *di[] = {din.c_str()}, *dO[] = {dout.c_str()};
        auto res = det->Run(Ort::RunOptions{nullptr}, di, &t, 1, dO, 1);
        const float *pm = res[0].GetTensorData<float>();
        std::vector<float> dbg; if (const char *e = getenv("ONLOC_PRED")) { dbg.resize((size_t) H * W); FILE *f = fopen(e, "rb"); if (fread(dbg.data(), 4, dbg.size(), f) != dbg.size()) fprintf(stderr, "short pred\n"); fclose(f); pm = dbg.data(); }
        auto t1 = clk::now(); R.t_det = ms(t0, t1);
        // DB post-processing on 8-connected components
        std::vector<int> lab((size_t) H * W, 0); std::vector<int> st; int nc = 0;
        std::vector<std::array<P, 4>> quads;
        double wsc = (double) iw / W, hsc = (double) ih / H;
        for (int i = 0; i < H * W && (int) quads.size() < 1000; i++) {
            if (pm[i] <= 0.3f || lab[i]) continue;
            ++nc; st.assign(1, i); lab[i] = nc; std::vector<P> bnd;
            while (!st.empty()) { int p = st.back(); st.pop_back(); int px = p % W, py = p / W; bool edge = false;
                for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) { if (!dx && !dy) continue;
                    int qx = px + dx, qy = py + dy;
                    if (qx < 0 || qy < 0 || qx >= W || qy >= H) { edge = true; continue; }
                    int q = qy * W + qx;
                    if (pm[q] > 0.3f) { if (!lab[q]) { lab[q] = nc; st.push_back(q); } } else if (!dx || !dy) edge = true; }
                if (edge) bnd.push_back({(double) px, (double) py}); }
            if (bnd.empty()) continue;
            RRect r = min_area_rect(bnd);
            if (std::min(r.w, r.h) < 3) continue;
            auto q = rect_points(r);
            // box_score_fast: mean of the map inside the (int-truncated) quad within its clamped bounding box
            std::array<P, 4> qi; for (int k = 0; k < 4; k++) qi[k] = {(double) (int) q[k].x, (double) (int) q[k].y};
            int xmin = std::clamp((int) std::floor(std::min({q[0].x, q[1].x, q[2].x, q[3].x})), 0, W - 1), xmax = std::clamp((int) std::ceil(std::max({q[0].x, q[1].x, q[2].x, q[3].x})), 0, W - 1);
            int ymin = std::clamp((int) std::floor(std::min({q[0].y, q[1].y, q[2].y, q[3].y})), 0, H - 1), ymax = std::clamp((int) std::ceil(std::max({q[0].y, q[1].y, q[2].y, q[3].y})), 0, H - 1);
            double s = 0; long n = 0;
            for (int yy = ymin; yy <= ymax; yy++) {
                double lo = 1e18, hi = -1e18;
                for (int k = 0; k < 4; k++) { const P &a = qi[k], &b = qi[(k + 1) % 4];
                    double ya = std::max(std::min(a.y, b.y), yy - 0.5), yb = std::min(std::max(a.y, b.y), yy + 0.5);
                    if (ya > yb) continue;
                    if (a.y == b.y) { lo = std::min({lo, a.x, b.x}); hi = std::max({hi, a.x, b.x}); continue; }
                    for (double y : {ya, yb}) { double xx = a.x + (y - a.y) * (b.x - a.x) / (b.y - a.y); lo = std::min(lo, xx); hi = std::max(hi, xx); } }
                if (lo > hi) continue;
                int x0 = std::max(xmin, (int) std::nearbyint(lo)), x1 = std::min(xmax, (int) std::nearbyint(hi));
                for (int xx = x0; xx <= x1; xx++) { s += pm[(size_t) yy * W + xx]; n++; }
            }
            if (getenv("ONLOC_DEBUG")) { double bx0=1e9,by0=1e9,bx1=-1,by1=-1; for (auto &p : bnd) { bx0=std::min(bx0,p.x); by0=std::min(by0,p.y); bx1=std::max(bx1,p.x); by1=std::max(by1,p.y); }
                fprintf(stderr, "comp %d %d %d %d rect %.2f %.2f score %.3f\n", (int) bx0, (int) by0, (int) (bx1-bx0+1), (int) (by1-by0+1), r.w, r.h, n ? s / n : 0); }
            if (!n || s / n < 0.6) continue;
            double area = r.w * r.h, per = 2 * (r.w + r.h), dist = area * 1.5 / per;
            // pyclipper works on integer coordinates: offset the truncated quad (round joins), then take its min-area rectangle
            std::vector<P> off;
            for (int k = 0; k < 4; k++) { const P &c = qi[k];
                for (int a = 0; a < 64; a++) { double th = a * M_PI / 32;
                    off.push_back({std::nearbyint(c.x + dist * std::cos(th)), std::nearbyint(c.y + dist * std::sin(th))}); } }
            RRect e = min_area_rect(off);
            if (std::min(e.w, e.h) < 5) continue;
            auto u = rect_points(e);
            for (auto &pt : u) { pt.x = std::clamp((double) std::lround(pt.x * wsc), 0.0, (double) iw); pt.y = std::clamp((double) std::lround(pt.y * hsc), 0.0, (double) ih); }
            quads.push_back(u);
        }
        // sort top-to-bottom, left-to-right (SortQuadBoxes)
        std::sort(quads.begin(), quads.end(), [](const auto &a, const auto &b) { return a[0].y < b[0].y || (a[0].y == b[0].y && a[0].x < b[0].x); });
        for (int i = 0; i + 1 < (int) quads.size(); i++)
            for (int j = i; j >= 0; j--) { if (std::abs(quads[j + 1][0].y - quads[j][0].y) < 10 && quads[j + 1][0].x < quads[j][0].x) std::swap(quads[j], quads[j + 1]); else break; }
        // rectified line crops: size, rotation, ratio
        struct Sub { int w, h; bool rot; double ratio; };
        std::vector<Sub> subs;
        for (auto &q : quads) {
            int cw = (int) std::max(std::hypot(q[0].x - q[1].x, q[0].y - q[1].y), std::hypot(q[2].x - q[3].x, q[2].y - q[3].y));
            int chh = (int) std::max(std::hypot(q[0].x - q[3].x, q[0].y - q[3].y), std::hypot(q[1].x - q[2].x, q[1].y - q[2].y));
            cw = std::max(cw, 1); chh = std::max(chh, 1);
            bool rot = (double) chh / cw >= 1.5;
            subs.push_back({cw, chh, rot, rot ? (double) chh / cw : (double) cw / chh});
        }
        std::vector<int> order(quads.size()); for (size_t i = 0; i < order.size(); i++) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return subs[a].ratio < subs[b].ratio; });
        std::vector<std::string> texts(quads.size()); std::vector<float> scores(quads.size());
        const float rm = 0.5f, rs = 0.5f;
        for (size_t i0 = 0; i0 < order.size(); i0 += 6) {
            size_t n = std::min<size_t>(6, order.size() - i0); double mr = 320.0 / 48;
            for (size_t j = 0; j < n; j++) mr = std::max(mr, subs[order[i0 + j]].ratio);
            int RW = std::min(3200, (int) (48 * mr));
            std::vector<float> r((size_t) n * 3 * 48 * RW, 0.f);
            for (size_t j = 0; j < n; j++) {
                const auto &q = quads[order[i0 + j]]; const Sub &s = subs[order[i0 + j]];
                auto cr = warp_cubic(img, iw, ih, q, s.w, s.h); int ow = s.w, oh = s.h;
                if (s.rot) { cr = rot90ccw(cr, s.w, s.h); ow = s.h; oh = s.w; }
                int rw = std::min(RW, (int) std::ceil(48.0 * ow / oh));
                auto rs8 = resize_linear(cr, ow, oh, rw, 48);
                for (int yy = 0; yy < 48; yy++) for (int xx = 0; xx < rw; xx++) for (int c = 0; c < 3; c++)
                    r[j * 3 * 48 * RW + (size_t) c * 48 * RW + (size_t) yy * RW + xx] = (rs8[((size_t) yy * rw + xx) * 3 + (2 - c)] / 255.f - rm) / rs;   // BGR
            }
            std::array<int64_t, 4> rsh{(int64_t) n, 3, 48, RW};
            Ort::Value rt = Ort::Value::CreateTensor<float>(mem, r.data(), r.size(), rsh.data(), 4);
            const char *ri[] = {rin.c_str()}, *ro[] = {rout.c_str()};
            auto rr = rec->Run(Ort::RunOptions{nullptr}, ri, &rt, 1, ro, 1);
            auto shp = rr[0].GetTensorTypeAndShapeInfo().GetShape(); const float *o = rr[0].GetTensorData<float>();
            for (size_t j = 0; j < n; j++) {
                std::string txt; double sum = 0; int cnt = 0, prev = -1;
                for (int64_t tt = 0; tt < shp[1]; tt++) {
                    const float *row = o + (j * shp[1] + tt) * shp[2];
                    int k = std::max_element(row, row + shp[2]) - row;
                    if (k != 0 && k != prev && k < (int) dict.size()) { txt += dict[k]; sum += row[k]; cnt++; }
                    prev = k;
                }
                texts[order[i0 + j]] = txt; scores[order[i0 + j]] = cnt ? sum / cnt : 0.f;
            }
        }
        auto t2 = clk::now(); R.t_rec = ms(t1, t2);
        for (size_t i = 0; i < quads.size(); i++) {
            Line L; L.text = texts[i]; L.score = scores[i]; L.quad = quads[i];
            double x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
            for (auto &p : quads[i]) { x0 = std::min(x0, p.x); y0 = std::min(y0, p.y); x1 = std::max(x1, p.x); y1 = std::max(y1, p.y); }
            L.box = {(int) x0, (int) y0, (int) x1, (int) y1};
            R.lines.push_back(L);
        }
        locate(R.lines, iw, ih, R.crop, R.level);
        R.date = ocr_date(R.lines, &R.conf);
        if (want_crop) R.jpeg = crop_x2_jpeg(img, iw, R.crop);
        R.t_post = ms(t2, clk::now());
        return R;
    }
};

}  // namespace onloc
