// cuda_runner.cpp -- minimal CUDA Driver API executor for cuda-stress.
//
// Deliberately tiny and dependency-free (no CUDA runtime, no libcuda JSON parser):
// it loads a CUBIN, launches `test_kernel`, copies input, synchronizes, copies output
// back inside guard regions and writes a self-describing result blob plus a JSON report.
//
// The runner is NOT the pipeline controller. Python owns scheduling; this binary only
// executes one kernel launch. It exits non-zero on any failure so a poisoned CUDA
// context never contaminates the next task (Python spawns a fresh process per run).
//
// Result blob layout (little-endian):
//   [0..7]   magic          0x43554345_53545245 ("CUCESTRE")
//   [8..15]  test_id_hash   low 64 bits of FNV-1a(test_id) (Python-side hashes are full SHA-256)
//   [16..19] status         0=OK 1=CUDA_ERROR 2=OOB_WRITE 3=BAD_SPEC 4=TIMEOUT
//   [20..23] reserved
//   [24..31] meaningful_words
//   [32..39] canary         0xDEADBEEFCAFEBABE
//   [40..]   meaningful_words * uint64_t
//   tail     tail_canary    0xCAFEBABEDEADBEEF
//
// Exit codes: 0 OK, 1 CUDA error, 2 OOB write, 3 bad spec, 124 timeout.

#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr uint64_t kMagic      = 0x4355434553545245ULL; // "CUCESTRE"
constexpr uint64_t kCanary     = 0xDEADBEEFCAFEBABEULL;
constexpr uint64_t kTailCanary = 0xCAFEBABEDEADBEEFULL;

// Device-buffer guard sentinels (placed around the meaningful region on the device).
constexpr uint64_t kGuardA    = 0x5A5A5A5A5A5A5A5AULL;
constexpr uint64_t kGuardB    = 0xA5A5A5A5A5A5A5A5ULL;

enum Status : uint32_t { ST_OK = 0, ST_CUDA_ERROR = 1, ST_OOB_WRITE = 2, ST_BAD_SPEC = 3, ST_TIMEOUT = 4 };

std::atomic<bool> g_done{false};
std::atomic<bool> g_timed_out{false};
int g_timeout_ms = 60000;

void watchdog() {
    for (int waited = 0; waited < g_timeout_ms && !g_done.load(std::memory_order_acquire);
         waited += 10)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (g_done.load(std::memory_order_acquire)) return;
    g_timed_out = true;
    fprintf(stderr, "[runner] watchdog: execution exceeded %d ms, aborting\n", g_timeout_ms);
    fflush(stderr);
    _Exit(ST_TIMEOUT);
}

bool parse_uint(const char* s, unsigned long long* out) {
    if (!s || !*s) return false;
    char* end = nullptr;
    unsigned long long v = strtoull(s, &end, 0);
    if (end == s || *end != '\0') return false;
    *out = v;
    return true;
}

bool parse_dims(const char* s, unsigned dims[3]) {
    dims[0] = dims[1] = dims[2] = 1;
    if (!s || !*s) return false;
    std::string str(s);
    size_t p0 = str.find(',');
    std::string a = (p0 == std::string::npos) ? str : str.substr(0, p0);
    unsigned long long v;
    if (!parse_uint(a.c_str(), &v)) return false;
    dims[0] = (unsigned)v;
    if (p0 != std::string::npos) {
        size_t p1 = str.find(',', p0 + 1);
        std::string b = (p1 == std::string::npos) ? str.substr(p0 + 1) : str.substr(p0 + 1, p1 - p0 - 1);
        if (!parse_uint(b.c_str(), &v)) return false;
        dims[1] = (unsigned)v;
        if (p1 != std::string::npos) {
            std::string c = str.substr(p1 + 1);
            if (!parse_uint(c.c_str(), &v)) return false;
            dims[2] = (unsigned)v;
        }
    }
    return true;
}

std::vector<uint8_t> read_file(const char* path, size_t expect_bytes, bool* ok) {
    *ok = false;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[runner] cannot open %s\n", path); return {}; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0 || (expect_bytes != 0 && (size_t)sz != expect_bytes)) {
        fprintf(stderr, "[runner] %s size %ld, expected %zu\n", path, sz, expect_bytes);
        fclose(f);
        return {};
    }
    std::vector<uint8_t> buf((size_t)sz);
    size_t got = fread(buf.data(), 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) { fprintf(stderr, "[runner] short read on %s\n", path); return {}; }
    *ok = true;
    return buf;
}

bool write_file(const char* path, const void* data, size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    bool ok = fwrite(data, 1, n, f) == n;
    fclose(f);
    return ok;
}

std::string cu_err(CUresult r) {
    const char* name = "?";
    const char* desc = "?";
    cuGetErrorName(r, &name);
    cuGetErrorString(r, &desc);
    return std::string(name) + " (" + desc + ")";
}

uint64_t fnv1a(const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

std::string fnv1a_hex(const std::string& s) {
    char out[32];
    snprintf(out, sizeof(out), "%016llx", (unsigned long long)fnv1a(s.data(), s.size()));
    return out;
}

} // namespace

int main(int argc, char** argv) {
    const char* cubin_path = nullptr;
    const char* kernel_name = "test_kernel";
    const char* input_path = nullptr;
    const char* output_path = nullptr;
    const char* result_path = nullptr;
    const char* test_id = "";
    unsigned long long input_words = 0, output_words = 0, seed = 0, guards = 8, device = 0;
    unsigned grid[3] = {1, 1, 1};
    unsigned block[3] = {64, 1, 1};

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "[runner] missing value for %s\n", what); exit(ST_BAD_SPEC); }
            return argv[++i];
        };
        if (a == "--cubin") cubin_path = next("--cubin");
        else if (a == "--kernel") kernel_name = next("--kernel");
        else if (a == "--input") input_path = next("--input");
        else if (a == "--output") output_path = next("--output");
        else if (a == "--result") result_path = next("--result");
        else if (a == "--test-id") test_id = next("--test-id");
        else if (a == "--input-words") { if (!parse_uint(next("--input-words"), &input_words)) return ST_BAD_SPEC; }
        else if (a == "--output-words") { if (!parse_uint(next("--output-words"), &output_words)) return ST_BAD_SPEC; }
        else if (a == "--seed") { if (!parse_uint(next("--seed"), &seed)) return ST_BAD_SPEC; }
        else if (a == "--guards") { if (!parse_uint(next("--guards"), &guards)) return ST_BAD_SPEC; }
        else if (a == "--device") { if (!parse_uint(next("--device"), &device)) return ST_BAD_SPEC; }
        else if (a == "--timeout-ms") {
            unsigned long long t;
            if (!parse_uint(next("--timeout-ms"), &t)) return ST_BAD_SPEC;
            g_timeout_ms = (int)t;
        }
        else if (a == "--grid") { if (!parse_dims(next("--grid"), grid)) return ST_BAD_SPEC; }
        else if (a == "--block") { if (!parse_dims(next("--block"), block)) return ST_BAD_SPEC; }
        else { fprintf(stderr, "[runner] unknown arg %s\n", a.c_str()); return ST_BAD_SPEC; }
    }

    if (!cubin_path || !input_path || !output_path || !result_path || input_words == 0 || output_words == 0) {
        fprintf(stderr, "[runner] missing required args (--cubin --input --output --result --input-words --output-words)\n");
        return ST_BAD_SPEC;
    }

    auto t0 = std::chrono::steady_clock::now();
    auto elapsed_us = [&]() {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    };

    CUresult rc;
    rc = cuInit(0);
    if (rc != CUDA_SUCCESS) { fprintf(stderr, "[runner] cuInit failed: %s\n", cu_err(rc).c_str()); return ST_CUDA_ERROR; }

    int drv_version = 0;
    cuDriverGetVersion(&drv_version);
    char devname[256] = "";
    int cc_major = 0, cc_minor = 0, sm_count = 0;
    size_t devmem = 0;
    {
        CUdevice dev = 0;
        rc = cuDeviceGet(&dev, (int)device);
        if (rc != CUDA_SUCCESS) { fprintf(stderr, "[runner] cuDeviceGet(%llu) failed: %s\n", (unsigned long long)device, cu_err(rc).c_str()); return ST_CUDA_ERROR; }
        cuDeviceGetName(devname, sizeof(devname), dev);
        cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
        cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
        cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
        cuDeviceTotalMem(&devmem, dev);
    }

    CUcontext ctx = nullptr;
    rc = cuCtxCreate(&ctx, CU_CTX_SCHED_BLOCKING_SYNC, (CUdevice)device);
    if (rc != CUDA_SUCCESS) { fprintf(stderr, "[runner] cuCtxCreate failed: %s\n", cu_err(rc).c_str()); return ST_CUDA_ERROR; }

    bool ok;
    std::vector<uint8_t> cubin = read_file(cubin_path, 0, &ok);
    if (!ok) { cuCtxDestroy(ctx); return ST_BAD_SPEC; }

    CUmodule mod = nullptr;
    rc = cuModuleLoadData(&mod, cubin.data());
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[runner] cuModuleLoadData failed: %s\n", cu_err(rc).c_str());
        cuCtxDestroy(ctx);
        return ST_CUDA_ERROR;
    }
    CUfunction func = nullptr;
    rc = cuModuleGetFunction(&func, mod, kernel_name);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[runner] cuModuleGetFunction(%s) failed: %s\n", kernel_name, cu_err(rc).c_str());
        cuModuleUnload(mod);
        cuCtxDestroy(ctx);
        return ST_CUDA_ERROR;
    }

    size_t meaningful_bytes = output_words * sizeof(uint64_t);
    size_t out_total_words = output_words + guards * 2;

    std::vector<uint8_t> input = read_file(input_path, input_words * sizeof(uint64_t), &ok);
    if (!ok) { cuModuleUnload(mod); cuCtxDestroy(ctx); return ST_BAD_SPEC; }

    CUdeviceptr d_in = 0, d_out = 0;
    rc = cuMemAlloc(&d_in, input.size());
    if (rc != CUDA_SUCCESS) { fprintf(stderr, "[runner] cuMemAlloc in failed: %s\n", cu_err(rc).c_str()); cuModuleUnload(mod); cuCtxDestroy(ctx); return ST_CUDA_ERROR; }
    rc = cuMemAlloc(&d_out, out_total_words * sizeof(uint64_t));
    if (rc != CUDA_SUCCESS) { fprintf(stderr, "[runner] cuMemAlloc out failed: %s\n", cu_err(rc).c_str()); cuMemFree(d_in); cuModuleUnload(mod); cuCtxDestroy(ctx); return ST_CUDA_ERROR; }

    std::vector<uint64_t> out_init(out_total_words, kGuardB);
    for (size_t i = 0; i < guards; i++) out_init[i] = kGuardA;
    rc = cuMemcpyHtoD(d_out, out_init.data(), out_init.size() * sizeof(uint64_t));
    if (rc != CUDA_SUCCESS) { fprintf(stderr, "[runner] guard init failed: %s\n", cu_err(rc).c_str()); }

    rc = cuMemcpyHtoD(d_in, input.data(), input.size());
    if (rc != CUDA_SUCCESS) { fprintf(stderr, "[runner] input copy failed: %s\n", cu_err(rc).c_str()); cuMemFree(d_in); cuMemFree(d_out); cuModuleUnload(mod); cuCtxDestroy(ctx); return ST_CUDA_ERROR; }

    // Kernel parameter buffer: (const u64* input, u64* output, u32 input_words, u64 seed)
    // `output` must point at the START of the meaningful region, i.e. past the leading guard.
    CUdeviceptr d_out_meaningful = d_out + guards * sizeof(uint64_t);
    unsigned int in_words32 = (unsigned int)input_words;
    void* kernelParams[] = {
        (void*)&d_in, (void*)&d_out_meaningful, (void*)&in_words32, (void*)&seed,
    };

    std::thread wd(watchdog);
    rc = cuLaunchKernel(func, grid[0], grid[1], grid[2], block[0], block[1], block[2],
                        0, nullptr, kernelParams, nullptr);
    if (rc == CUDA_SUCCESS) rc = cuCtxSynchronize();
    g_done.store(true, std::memory_order_release);
    wd.detach();

    bool sync_ok = (rc == CUDA_SUCCESS);
    uint32_t status = sync_ok ? ST_OK : ST_CUDA_ERROR;

    std::vector<uint64_t> out_back(out_total_words, 0);
    if (sync_ok) {
        rc = cuMemcpyDtoH(out_back.data(), d_out, out_back.size() * sizeof(uint64_t));
        if (rc != CUDA_SUCCESS) { sync_ok = false; status = ST_CUDA_ERROR; }
    }

    bool guard_ok = sync_ok;
    if (sync_ok) {
        for (size_t i = 0; i < guards; i++)
            if (out_back[i] != kGuardA) guard_ok = false;
        for (size_t i = 0; i < guards; i++)
            if (out_back[guards + output_words + i] != kGuardB) guard_ok = false;
        if (!guard_ok) status = ST_OOB_WRITE;
    }

    std::vector<uint8_t> blob;
    auto append = [&](const void* d, size_t n) {
        const uint8_t* p = (const uint8_t*)d;
        blob.insert(blob.end(), p, p + n);
    };
    uint64_t test_hash = fnv1a(test_id, strlen(test_id));
    uint64_t mw = output_words;
    uint32_t st32 = status;
    uint32_t rsv = 0;
    append(&kMagic, 8);
    append(&test_hash, 8);
    append(&st32, 4);
    append(&rsv, 4);
    append(&mw, 8);
    append(&kCanary, 8);
    if (sync_ok) append(out_back.data() + guards, meaningful_bytes);
    append(&kTailCanary, 8);
    bool wrote = write_file(output_path, blob.data(), blob.size());

    std::string out_hex;
    if (sync_ok) out_hex = fnv1a_hex(std::string((const char*)(out_back.data() + guards), meaningful_bytes));
    std::string err_str = (rc == CUDA_SUCCESS) ? "" : cu_err(rc);
    std::string status_str = (status == ST_OK) ? "OK" :
                             (status == ST_CUDA_ERROR) ? "CUDA_ERROR" :
                             (status == ST_OOB_WRITE) ? "OUT_OF_BOUNDS_WRITE" :
                             (status == ST_TIMEOUT) ? "TIMEOUT" : "BAD_SPEC";

    char jbuf[4096];
    snprintf(jbuf, sizeof(jbuf),
        "{\n"
        "  \"status\": \"%s\",\n"
        "  \"status_code\": %u,\n"
        "  \"error\": \"%s\",\n"
        "  \"guard_ok\": %s,\n"
        "  \"elapsed_us\": %lld,\n"
        "  \"output_hash\": \"%s\",\n"
        "  \"meaningful_words\": %llu,\n"
        "  \"device\": \"%s\",\n"
        "  \"driver_version\": %d.%d,\n"
        "  \"sm\": \"%d%d\",\n"
        "  \"sm_count\": %d,\n"
        "  \"device_mem_mb\": %llu,\n"
        "  \"blob_written\": %s\n"
        "}\n",
        status_str.c_str(), status,
        err_str.c_str(),
        guard_ok ? "true" : "false",
        (long long)elapsed_us(),
        out_hex.c_str(),
        (unsigned long long)output_words,
        devname,
        drv_version / 1000, (drv_version % 100) / 10,
        cc_major, cc_minor,
        sm_count,
        (unsigned long long)(devmem >> 20),
        wrote ? "true" : "false");
    bool wrote_result = write_file(result_path, jbuf, strlen(jbuf));

    cuMemFree(d_in);
    cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);

    if (g_timed_out.load()) return ST_TIMEOUT;
    if (!sync_ok) return ST_CUDA_ERROR;
    if (!guard_ok) return ST_OOB_WRITE;
    if (!wrote || !wrote_result) return ST_CUDA_ERROR;
    return ST_OK;
}
