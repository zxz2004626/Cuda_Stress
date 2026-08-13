// cuda_runner.cpp -- minimal CUDA Driver API executor for cuda-stress.
//
// Deliberately dependency-free (no CUDA runtime, no libcuda JSON parser): it retains
// the fixed `test_kernel` ABI and also supports generator_ir's manifest-driven ABI.
// Both paths load a CUBIN, launch once, check guarded outputs, and write a
// self-describing result blob plus a JSON report.
//
// Python owns scheduling and timeouts.  In --server mode this process keeps one
// CUcontext across successful jobs and exits after the first failure, preventing a
// poisoned context from reaching the next task.
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
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
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

struct DeviceInfo {
    char name[256] = "";
    int driver_version = 0;
    int cc_major = 0;
    int cc_minor = 0;
    int sm_count = 0;
    size_t memory = 0;
};

struct IrArg {
    std::string kind;
    std::vector<uint8_t> data;
    size_t element_bytes = 0;
};

struct IrGlobal {
    std::string init;
    std::string name;
};

struct IrSpec {
    std::string cubin_path;
    std::string kernel_name;
    std::string output_path;
    std::string result_path;
    std::string test_id;
    std::string input_dist = "hash";
    unsigned grid[3] = {1, 1, 1};
    unsigned block[3] = {1, 1, 1};
    unsigned long long shared_bytes = 0;
    unsigned long long n = 0;
    unsigned long long buf_size = 0;
    unsigned long long in_size = 0;
    unsigned long long input_salt = 0;
    unsigned long long guards = 8;
    std::vector<IrArg> args;
    std::vector<IrGlobal> globals;
};

bool decode_hex(const std::string& text, std::vector<uint8_t>* out) {
    if (text.empty() || (text.size() & 1)) return false;
    out->clear();
    out->reserve(text.size() / 2);
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < text.size(); i += 2) {
        int hi = nibble(text[i]), lo = nibble(text[i + 1]);
        if (hi < 0 || lo < 0) return false;
        out->push_back((uint8_t)((hi << 4) | lo));
    }
    return true;
}

bool load_ir_spec(const char* path, IrSpec* spec, std::string* error) {
    std::ifstream in(path);
    if (!in) { *error = std::string("cannot open IR spec ") + path; return false; }
    std::string line;
    unsigned line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos) {
            *error = "IR spec line " + std::to_string(line_no) + " has no '='";
            return false;
        }
        std::string key = line.substr(0, eq), value = line.substr(eq + 1);
        unsigned long long number = 0;
        if (key == "version") {
            if (value != "1") { *error = "unsupported IR spec version"; return false; }
        }
        else if (key == "cubin") spec->cubin_path = value;
        else if (key == "kernel") spec->kernel_name = value;
        else if (key == "output") spec->output_path = value;
        else if (key == "result") spec->result_path = value;
        else if (key == "test_id") spec->test_id = value;
        else if (key == "input_dist") spec->input_dist = value;
        else if (key == "grid") {
            if (!parse_dims(value.c_str(), spec->grid)) { *error = "bad grid"; return false; }
        } else if (key == "block") {
            if (!parse_dims(value.c_str(), spec->block)) { *error = "bad block"; return false; }
        } else if (key == "shared_bytes" || key == "n" || key == "buf_size" ||
                   key == "in_size" || key == "input_salt" || key == "guards") {
            if (!parse_uint(value.c_str(), &number)) { *error = "bad integer for " + key; return false; }
            if (key == "shared_bytes") spec->shared_bytes = number;
            else if (key == "n") spec->n = number;
            else if (key == "buf_size") spec->buf_size = number;
            else if (key == "in_size") spec->in_size = number;
            else if (key == "input_salt") spec->input_salt = number;
            else spec->guards = number;
        } else if (key == "arg") {
            IrArg arg;
            size_t colon = value.find(':');
            arg.kind = value.substr(0, colon);
            std::string payload = colon == std::string::npos ? "" : value.substr(colon + 1);
            if (arg.kind == "scalar") {
                if (!decode_hex(payload, &arg.data)) { *error = "bad scalar hex"; return false; }
            } else if (arg.kind == "output_fp") {
                if (!parse_uint(payload.c_str(), &number) || (number != 4 && number != 8)) {
                    *error = "bad output_fp element size"; return false;
                }
                arg.element_bytes = (size_t)number;
            } else if (arg.kind != "output" && arg.kind != "input" && arg.kind != "n") {
                *error = "unknown arg kind " + arg.kind; return false;
            }
            spec->args.push_back(std::move(arg));
        } else if (key == "global") {
            size_t colon = value.find(':');
            if (colon == std::string::npos) { *error = "bad global spec"; return false; }
            IrGlobal global{value.substr(0, colon), value.substr(colon + 1)};
            if (global.init != "zero" && global.init != "cmem_u32") {
                *error = "unknown global initializer " + global.init; return false;
            }
            spec->globals.push_back(std::move(global));
        } else {
            *error = "unknown IR spec key " + key;
            return false;
        }
    }
    if (spec->cubin_path.empty() || spec->kernel_name.empty() ||
        spec->output_path.empty() || spec->result_path.empty() ||
        spec->n == 0 || spec->buf_size == 0 || spec->in_size == 0 || spec->args.empty()) {
        *error = "IR spec is missing required fields";
        return false;
    }
    return true;
}

bool init_driver_context(unsigned long long device, CUcontext* ctx, DeviceInfo* info) {
    CUresult rc = cuInit(0);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[runner] cuInit failed: %s\n", cu_err(rc).c_str());
        return false;
    }
    CUdevice dev = 0;
    rc = cuDeviceGet(&dev, (int)device);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[runner] cuDeviceGet(%llu) failed: %s\n", device, cu_err(rc).c_str());
        return false;
    }
    cuDriverGetVersion(&info->driver_version);
    cuDeviceGetName(info->name, sizeof(info->name), dev);
    cuDeviceGetAttribute(&info->cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&info->cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cuDeviceGetAttribute(&info->sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    cuDeviceTotalMem(&info->memory, dev);
    rc = cuCtxCreate(ctx, CU_CTX_SCHED_BLOCKING_SYNC, dev);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[runner] cuCtxCreate failed: %s\n", cu_err(rc).c_str());
        return false;
    }
    return true;
}

uint32_t input_word(const std::string& dist, uint32_t i) {
    if (dist == "zeros") return 0;
    if (dist == "small") return i % 17u;
    if (dist == "random") return (i * 1103515245u + 12345u) ^ (i >> 3);
    if (dist == "adversarial")
        return ((i & 1u) ? 0xFFFFFFFFu : 0u) ^ ((i & 2u) ? 0x80000000u : 0u);
    return i * 2654435761u + 12345u;
}

void apply_input_salt(std::vector<uint32_t>* input, uint64_t salt) {
    if (salt == 0) return;
    for (size_t i = 0; i < input->size(); ++i) {
        uint64_t z = salt + 0x9e3779b97f4a7c15ULL * ((uint64_t)(uint32_t)i + 1ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z ^= z >> 31;
        uint32_t salt_word = (uint32_t)(z ^ (z >> 32));
        (*input)[i] ^= salt_word ? salt_word : 0xa5a5a5a5u;
    }
}

bool all_bytes(const std::vector<uint8_t>& data, size_t begin, size_t count, uint8_t value) {
    if (begin > data.size() || count > data.size() - begin) return false;
    return std::all_of(data.begin() + begin, data.begin() + begin + count,
                       [value](uint8_t x) { return x == value; });
}

int run_ir_job(CUcontext ctx, const DeviceInfo& info, const char* spec_path,
               bool context_reused) {
    (void)ctx;
    auto started = std::chrono::steady_clock::now();
    auto since_us = [&](std::chrono::steady_clock::time_point point) -> long long {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - point).count();
    };
    IrSpec spec;
    std::string parse_error;
    if (!load_ir_spec(spec_path, &spec, &parse_error)) {
        fprintf(stderr, "[runner] %s\n", parse_error.c_str());
        return ST_BAD_SPEC;
    }

    CUmodule mod = nullptr;
    CUfunction func = nullptr;
    CUdeviceptr d_in = 0, d_out = 0, d_fp = 0;
    size_t fp_element_bytes = 0, lead_fp_bytes = 0;
    auto cleanup = [&]() {
        if (d_fp) cuMemFree(d_fp);
        if (d_in) cuMemFree(d_in);
        if (d_out) cuMemFree(d_out);
        if (mod) cuModuleUnload(mod);
    };
    auto write_error = [&](uint32_t status, const std::string& message) -> int {
        char jbuf[4096];
        snprintf(jbuf, sizeof(jbuf),
            "{\n  \"status\": \"%s\",\n  \"status_code\": %u,\n"
            "  \"error\": \"%s\",\n  \"elapsed_us\": %lld,\n"
            "  \"context_reused\": %s\n}\n",
            status == ST_BAD_SPEC ? "BAD_SPEC" : "CUDA_ERROR", status,
            message.c_str(), since_us(started), context_reused ? "true" : "false");
        write_file(spec.result_path.c_str(), jbuf, strlen(jbuf));
        cleanup();
        return (int)status;
    };

    bool ok = false;
    std::vector<uint8_t> cubin = read_file(spec.cubin_path.c_str(), 0, &ok);
    if (!ok) return write_error(ST_BAD_SPEC, "cannot read cubin");
    auto module_started = std::chrono::steady_clock::now();
    CUresult rc = cuModuleLoadData(&mod, cubin.data());
    if (rc != CUDA_SUCCESS) return write_error(ST_CUDA_ERROR, cu_err(rc));
    long long module_load_us = since_us(module_started);
    rc = cuModuleGetFunction(&func, mod, spec.kernel_name.c_str());
    if (rc != CUDA_SUCCESS) return write_error(ST_CUDA_ERROR, cu_err(rc));

    for (const auto& global : spec.globals) {
        CUdeviceptr ptr = 0;
        size_t bytes = 0;
        rc = cuModuleGetGlobal(&ptr, &bytes, mod, global.name.c_str());
        if (rc != CUDA_SUCCESS)
            return write_error(ST_CUDA_ERROR, "cuModuleGetGlobal(" + global.name + "): " + cu_err(rc));
        if (global.init == "zero") {
            rc = cuMemsetD8(ptr, 0, bytes);
        } else {
            if (bytes == 0 || (bytes % sizeof(uint32_t)) != 0)
                return write_error(ST_BAD_SPEC, "cmem_u32 global has invalid size");
            std::vector<uint32_t> values(bytes / sizeof(uint32_t));
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = (uint32_t)i * 2654435761u + 40503u;
            rc = cuMemcpyHtoD(ptr, values.data(), bytes);
        }
        if (rc != CUDA_SUCCESS) return write_error(ST_CUDA_ERROR, cu_err(rc));
    }

    std::vector<uint32_t> input((size_t)spec.in_size);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = input_word(spec.input_dist, (uint32_t)i);
    apply_input_salt(&input, spec.input_salt);

    rc = cuMemAlloc(&d_in, input.size() * sizeof(uint32_t));
    if (rc != CUDA_SUCCESS) return write_error(ST_CUDA_ERROR, cu_err(rc));
    rc = cuMemcpyHtoD(d_in, input.data(), input.size() * sizeof(uint32_t));
    if (rc != CUDA_SUCCESS) return write_error(ST_CUDA_ERROR, cu_err(rc));

    // Preserve cudaMalloc's >=256-byte alignment for the meaningful pointer.
    size_t requested_guard_bytes = (size_t)spec.guards * sizeof(uint32_t);
    size_t lead_bytes = requested_guard_bytes == 0 ? 0 :
        ((std::max<size_t>(requested_guard_bytes, 256) + 255) / 256) * 256;
    size_t meaningful_bytes = (size_t)spec.buf_size * sizeof(uint32_t);
    size_t out_total_bytes = lead_bytes + meaningful_bytes + lead_bytes;
    rc = cuMemAlloc(&d_out, out_total_bytes);
    if (rc != CUDA_SUCCESS) return write_error(ST_CUDA_ERROR, cu_err(rc));
    if (lead_bytes) rc = cuMemsetD8(d_out, 0x5A, lead_bytes);
    if (rc == CUDA_SUCCESS) rc = cuMemsetD8(d_out + lead_bytes, 0, meaningful_bytes);
    if (rc == CUDA_SUCCESS && lead_bytes)
        rc = cuMemsetD8(d_out + lead_bytes + meaningful_bytes, 0xA5, lead_bytes);
    if (rc != CUDA_SUCCESS) return write_error(ST_CUDA_ERROR, cu_err(rc));
    CUdeviceptr d_out_meaningful = d_out + lead_bytes;

    for (const auto& arg : spec.args)
        if (arg.kind == "output_fp") fp_element_bytes = arg.element_bytes;
    CUdeviceptr d_fp_meaningful = 0;
    if (fp_element_bytes) {
        size_t requested = (size_t)spec.guards * fp_element_bytes;
        lead_fp_bytes = requested == 0 ? 0 :
            ((std::max<size_t>(requested, 256) + 255) / 256) * 256;
        size_t fp_meaningful = (size_t)spec.buf_size * fp_element_bytes;
        rc = cuMemAlloc(&d_fp, lead_fp_bytes + fp_meaningful + lead_fp_bytes);
        if (rc != CUDA_SUCCESS) return write_error(ST_CUDA_ERROR, cu_err(rc));
        if (lead_fp_bytes) rc = cuMemsetD8(d_fp, 0x5A, lead_fp_bytes);
        if (rc == CUDA_SUCCESS) rc = cuMemsetD8(d_fp + lead_fp_bytes, 0, fp_meaningful);
        if (rc == CUDA_SUCCESS && lead_fp_bytes)
            rc = cuMemsetD8(d_fp + lead_fp_bytes + fp_meaningful, 0xA5, lead_fp_bytes);
        if (rc != CUDA_SUCCESS) return write_error(ST_CUDA_ERROR, cu_err(rc));
        d_fp_meaningful = d_fp + lead_fp_bytes;
    }

    int n_arg = (int)spec.n;
    std::vector<std::vector<uint8_t>> scalar_storage;
    scalar_storage.reserve(spec.args.size());
    std::vector<void*> kernel_params;
    kernel_params.reserve(spec.args.size());
    for (const auto& arg : spec.args) {
        if (arg.kind == "output") kernel_params.push_back(&d_out_meaningful);
        else if (arg.kind == "input") kernel_params.push_back(&d_in);
        else if (arg.kind == "n") kernel_params.push_back(&n_arg);
        else if (arg.kind == "output_fp") kernel_params.push_back(&d_fp_meaningful);
        else {
            scalar_storage.push_back(arg.data);
            kernel_params.push_back(scalar_storage.back().data());
        }
    }

    auto kernel_started = std::chrono::steady_clock::now();
    rc = cuLaunchKernel(func,
                        spec.grid[0], spec.grid[1], spec.grid[2],
                        spec.block[0], spec.block[1], spec.block[2],
                        (unsigned)spec.shared_bytes, nullptr,
                        kernel_params.data(), nullptr);
    if (rc == CUDA_SUCCESS) rc = cuCtxSynchronize();
    long long kernel_us = since_us(kernel_started);
    if (rc != CUDA_SUCCESS) return write_error(ST_CUDA_ERROR, cu_err(rc));

    std::vector<uint8_t> out_back(out_total_bytes);
    rc = cuMemcpyDtoH(out_back.data(), d_out, out_back.size());
    if (rc != CUDA_SUCCESS) return write_error(ST_CUDA_ERROR, cu_err(rc));
    bool guard_ok = !lead_bytes ||
        (all_bytes(out_back, 0, lead_bytes, 0x5A) &&
         all_bytes(out_back, lead_bytes + meaningful_bytes, lead_bytes, 0xA5));
    if (guard_ok && d_fp && lead_fp_bytes) {
        std::vector<uint8_t> fp_guard(lead_fp_bytes);
        rc = cuMemcpyDtoH(fp_guard.data(), d_fp, lead_fp_bytes);
        if (rc == CUDA_SUCCESS)
            guard_ok = all_bytes(fp_guard, 0, lead_fp_bytes, 0x5A);
        if (rc == CUDA_SUCCESS) {
            size_t fp_meaningful = (size_t)spec.buf_size * fp_element_bytes;
            rc = cuMemcpyDtoH(fp_guard.data(), d_fp + lead_fp_bytes + fp_meaningful,
                              lead_fp_bytes);
        }
        if (rc == CUDA_SUCCESS)
            guard_ok = guard_ok && all_bytes(fp_guard, 0, lead_fp_bytes, 0xA5);
        if (rc != CUDA_SUCCESS) return write_error(ST_CUDA_ERROR, cu_err(rc));
    }

    const uint32_t* output = reinterpret_cast<const uint32_t*>(out_back.data() + lead_bytes);
    uint64_t hash = 42;
    for (size_t i = 0; i < (size_t)spec.buf_size; ++i) {
        uint64_t val = output[i];
        hash ^= val + 0x9e3779b9ULL + (hash << 6) + (hash >> 2);
    }
    uint32_t status = guard_ok ? ST_OK : ST_OOB_WRITE;
    std::vector<uint8_t> blob;
    auto append = [&](const void* data, size_t bytes) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        blob.insert(blob.end(), p, p + bytes);
    };
    uint64_t test_hash = fnv1a(spec.test_id.data(), spec.test_id.size());
    uint32_t reserved = 0;
    uint64_t words = 1;
    append(&kMagic, 8); append(&test_hash, 8); append(&status, 4); append(&reserved, 4);
    append(&words, 8); append(&kCanary, 8);
    if (status == ST_OK) append(&hash, 8);
    append(&kTailCanary, 8);
    bool blob_written = write_file(spec.output_path.c_str(), blob.data(), blob.size());

    char jbuf[4096];
    snprintf(jbuf, sizeof(jbuf),
        "{\n  \"status\": \"%s\",\n  \"status_code\": %u,\n"
        "  \"error\": \"\",\n  \"guard_ok\": %s,\n"
        "  \"elapsed_us\": %lld,\n  \"module_load_us\": %lld,\n"
        "  \"kernel_us\": %lld,\n  \"checksum\": %llu,\n"
        "  \"context_reused\": %s,\n  \"device\": \"%s\",\n"
        "  \"driver_version\": %d.%d,\n  \"sm\": \"%d%d\",\n"
        "  \"sm_count\": %d,\n  \"device_mem_mb\": %llu,\n"
        "  \"blob_written\": %s\n}\n",
        status == ST_OK ? "OK" : "OUT_OF_BOUNDS_WRITE", status,
        guard_ok ? "true" : "false", since_us(started), module_load_us, kernel_us,
        (unsigned long long)hash, context_reused ? "true" : "false", info.name,
        info.driver_version / 1000, (info.driver_version % 100) / 10,
        info.cc_major, info.cc_minor, info.sm_count,
        (unsigned long long)(info.memory >> 20), blob_written ? "true" : "false");
    bool result_written = write_file(spec.result_path.c_str(), jbuf, strlen(jbuf));
    cleanup();
    if (!blob_written || !result_written) return ST_CUDA_ERROR;
    return (int)status;
}

int run_ir_server(unsigned long long device) {
    CUcontext ctx = nullptr;
    DeviceInfo info;
    if (!init_driver_context(device, &ctx, &info)) return ST_CUDA_ERROR;
    printf("READY\n");
    fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "QUIT") break;
        if (line.rfind("RUN\t", 0) != 0) {
            fprintf(stderr, "[runner] bad server command\n");
            continue;
        }
        size_t second_tab = line.find('\t', 4);
        if (second_tab == std::string::npos) {
            fprintf(stderr, "[runner] malformed RUN command\n");
            continue;
        }
        std::string sequence = line.substr(4, second_tab - 4);
        std::string path = line.substr(second_tab + 1);
        int status = run_ir_job(ctx, info, path.c_str(), true);
        printf("DONE\t%s\t%d\n", sequence.c_str(), status);
        fflush(stdout);
        if (status != ST_OK) {
            cuCtxDestroy(ctx);
            return status;
        }
    }
    cuCtxDestroy(ctx);
    return ST_OK;
}

} // namespace

int main(int argc, char** argv) {
    bool server_mode = false;
    const char* ir_spec_path = nullptr;
    unsigned long long early_device = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--server") server_mode = true;
        else if (a == "--ir-spec" && i + 1 < argc) ir_spec_path = argv[++i];
        else if (a == "--device" && i + 1 < argc) {
            if (!parse_uint(argv[++i], &early_device)) return ST_BAD_SPEC;
        }
    }
    if (server_mode) return run_ir_server(early_device);
    if (ir_spec_path) {
        CUcontext ir_ctx = nullptr;
        DeviceInfo ir_info;
        if (!init_driver_context(early_device, &ir_ctx, &ir_info)) return ST_CUDA_ERROR;
        int status = run_ir_job(ir_ctx, ir_info, ir_spec_path, false);
        cuCtxDestroy(ir_ctx);
        return status;
    }

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
