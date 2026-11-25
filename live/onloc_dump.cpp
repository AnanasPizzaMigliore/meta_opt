// Run the on-device localisation port on a list of frames and dump, per frame: name, level, OCR date, crop box,
// line count, ms (detector, recogniser, rest), then the recognised lines ('\x1f'-separated). Optional crop dir.
//   onloc_dump DET REC DICT LIST OUT.tsv [CROPDIR] [THREADS]
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "onloc.hpp"
int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: onloc_dump DET REC DICT LIST OUT [CROPDIR] [THREADS]\n"); return 2; }
    std::string cropdir = argc > 6 ? argv[6] : ""; int th = argc > 7 ? atoi(argv[7]) : 4;
    onloc::Pipeline pl(argv[1], argv[2], argv[3], th);
    std::ifstream in(argv[4]); FILE *out = fopen(argv[5], "w"); std::string path;
    while (std::getline(in, path)) {
        if (path.empty()) continue;
        int iw, ih, ch; unsigned char *img = stbi_load(path.c_str(), &iw, &ih, &ch, 3);
        if (!img) { fprintf(stderr, "cannot read %s\n", path.c_str()); continue; }
        auto R = pl.run(img, iw, ih, !cropdir.empty());
        std::string name = path.substr(path.find_last_of('/') + 1);
        if (!cropdir.empty()) { FILE *f = fopen((cropdir + "/" + name).c_str(), "wb"); fwrite(R.jpeg.data(), 1, R.jpeg.size(), f); fclose(f); }
        fprintf(out, "%s\t%s\t%s\t%d,%d,%d,%d\t%zu\t%.0f\t%.0f\t%.0f\t", name.c_str(), R.level.c_str(), R.date.c_str(),
                R.crop[0], R.crop[1], R.crop[2], R.crop[3], R.lines.size(), R.t_det, R.t_rec, R.t_post);
        for (size_t i = 0; i < R.lines.size(); i++) fprintf(out, "%s%s|%.4f|%d,%d,%d,%d", i ? "\x1f" : "", R.lines[i].text.c_str(), R.lines[i].score,
                R.lines[i].box[0], R.lines[i].box[1], R.lines[i].box[2], R.lines[i].box[3]);
        fprintf(out, "\n"); fflush(out);
        stbi_image_free(img);
    }
    fclose(out); return 0;
}
