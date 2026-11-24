// Times one ONNX model on random input: ortbench MODEL N C H W THREADS REPS [xnnpack]
// Prints the median and min wall time per run (ms) after one warm-up run.
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
int main(int argc, char **argv) {
    if (argc < 8) { fprintf(stderr, "usage: ortbench MODEL N C H W THREADS REPS [xnnpack]\n"); return 2; }
    std::vector<int64_t> shape = {atoll(argv[2]), atoll(argv[3]), atoll(argv[4]), atoll(argv[5])};
    int threads = atoi(argv[6]), reps = atoi(argv[7]);
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bench");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(threads);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (argc > 8 && std::string(argv[8]) == "xnnpack")
        so.AppendExecutionProvider("XNNPACK", {{"intra_op_num_threads", std::to_string(threads)}});
    Ort::Session s(env, argv[1], so);
    Ort::AllocatorWithDefaultOptions alloc;
    auto in = s.GetInputNameAllocated(0, alloc);
    auto out = s.GetOutputNameAllocated(0, alloc);
    size_t n = 1; for (auto d : shape) n *= d;
    std::vector<float> x(n);
    for (auto &v : x) v = (float) rand() / RAND_MAX;
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value t = Ort::Value::CreateTensor<float>(mem, x.data(), n, shape.data(), shape.size());
    const char *ins[] = {in.get()}, *outs[] = {out.get()};
    std::vector<double> ms;
    for (int i = 0; i <= reps; i++) {
        auto t0 = std::chrono::steady_clock::now();
        s.Run(Ort::RunOptions{nullptr}, ins, &t, 1, outs, 1);
        double d = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (i) ms.push_back(d);
    }
    std::sort(ms.begin(), ms.end());
    printf("%s %lldx%lldx%lldx%lld t=%d %s median %.1f ms min %.1f ms\n", argv[1], (long long) shape[0], (long long) shape[1],
           (long long) shape[2], (long long) shape[3], threads, argc > 8 ? argv[8] : "cpu", ms[ms.size() / 2], ms[0]);
    return 0;
}
