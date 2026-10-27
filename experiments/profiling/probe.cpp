#include "common.h"
#include "llama.h"
#include "json.hpp"

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <nvToolsExt.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <time.h>
#include <vector>

using json = nlohmann::ordered_json;

static bool env_flag(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "1") == 0;
}

static std::string required_env(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        throw std::runtime_error(std::string("missing environment variable: ") + name);
    }
    return value;
}

static int64_t monotonic_ns() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        throw std::runtime_error("clock_gettime failed");
    }
    return int64_t(ts.tv_sec) * 1000000000LL + int64_t(ts.tv_nsec);
}

static void require_cuda(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

static void require_directory(const std::string & path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0 || !S_ISDIR(info.st_mode)) {
        throw std::runtime_error("LCP_OUT must be an existing directory: " + path);
    }
}

static void require_absent(const std::string & path) {
    struct stat info;
    if (lstat(path.c_str(), &info) == 0) {
        throw std::runtime_error("refusing to overwrite existing output: " + path);
    }
    if (errno != ENOENT) {
        throw std::runtime_error("cannot check output path: " + path);
    }
}

static void dump_f32(const std::string & path, const std::vector<float> & values) {
    std::ofstream out(path, std::ios::binary | std::ios::out);
    if (!out) {
        throw std::runtime_error("cannot open output: " + path);
    }
    out.write(reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
    out.close();
    if (!out) {
        throw std::runtime_error("cannot write output: " + path);
    }
}

static void dump_json(const std::string & path, const json & value) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot open JSON output: " + path);
    }
    out << value.dump(2) << '\n';
    out.close();
    if (!out) {
        throw std::runtime_error("cannot write JSON output: " + path);
    }
}

static std::vector<llama_token> get_tokens(const json & value, size_t count,
        int32_t n_vocab, const std::string & label) {
    if (!value.is_array() || value.size() != count) {
        throw std::runtime_error(label + " has incorrect token count");
    }
    std::vector<llama_token> tokens;
    tokens.reserve(count);
    for (const auto & item : value) {
        if (!item.is_number_integer()) {
            throw std::runtime_error(label + " contains a non-integer token");
        }
        const int64_t token = item.get<int64_t>();
        if (token < 0 || token >= n_vocab) {
            throw std::runtime_error(label + " contains an out-of-vocabulary token");
        }
        tokens.push_back(static_cast<llama_token>(token));
    }
    return tokens;
}

struct Batch {
    llama_batch value;
    explicit Batch(int32_t capacity) : value(llama_batch_init(capacity, 0, 1)) {}
    ~Batch() { llama_batch_free(value); }
    Batch(const Batch &) = delete;
    Batch & operator=(const Batch &) = delete;
};

struct CaptureRange {
    bool capture_active = false;
    bool nvtx_active = false;

    void begin(bool start_capture, bool nvtx, const std::string & name) {
        if (start_capture) {
            if (capture_active) {
                throw std::runtime_error("profiler capture was already active");
            }
            require_cuda(cudaProfilerStart(), "cudaProfilerStart");
            capture_active = true;
        }
        if (nvtx) {
            nvtxRangePushA(name.c_str());
            nvtx_active = true;
        }
    }

    void end(bool stop_capture) {
        if (nvtx_active) {
            nvtxRangePop();
            nvtx_active = false;
        }
        if (stop_capture && capture_active) {
            const cudaError_t status = cudaProfilerStop();
            capture_active = false;
            require_cuda(status, "cudaProfilerStop");
        }
    }

    ~CaptureRange() {
        if (nvtx_active) {
            nvtxRangePop();
        }
        if (capture_active) {
            cudaProfilerStop();
        }
    }
};

static void prepare_decode(llama_context * ctx, llama_batch batch) {
    const int rc = llama_decode(ctx, batch);
    if (rc != 0) {
        throw std::runtime_error("preparation llama_decode failed: " + std::to_string(rc));
    }
    llama_synchronize(ctx);
}

static void prepare_a(llama_context * ctx, const std::vector<llama_token> & prompt,
        const std::vector<llama_token> & generated) {
    llama_synchronize(ctx);
    llama_kv_cache_clear(ctx);
    {
        Batch batch(static_cast<int32_t>(prompt.size()));
        for (size_t i = 0; i < prompt.size(); ++i) {
            llama_batch_add(batch.value, prompt[i], static_cast<llama_pos>(i),
                    {0}, i + 1 == prompt.size(), 0);
        }
        prepare_decode(ctx, batch.value);
    }
    for (size_t i = 0; i < 11; ++i) {
        Batch batch(1);
        llama_batch_add(batch.value, generated[i], static_cast<llama_pos>(16 + i),
                {0}, true, 0);
        prepare_decode(ctx, batch.value);
    }
}

static std::vector<float> copy_finite(const float * ptr, size_t count,
        const std::string & label) {
    if (ptr == nullptr) {
        throw std::runtime_error("null output: " + label);
    }
    std::vector<float> values(ptr, ptr + count);
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("non-finite output: " + label);
        }
    }
    return values;
}

static json run_batch(llama_context * ctx, llama_model * model,
        const std::vector<llama_token> & a_prompt,
        const std::vector<llama_token> & a_generated,
        const std::vector<llama_token> & b_prompt,
        const std::string & root, bool measured, bool capture, bool nvtx,
        CaptureRange & range, bool first_measurement, bool last_measurement) {
    prepare_a(ctx, a_prompt, a_generated);
    const int32_t length = static_cast<int32_t>(b_prompt.size());
    const std::string tag = "B" + std::to_string(length);
    Batch batch(length + 1);
    llama_batch_add(batch.value, a_generated[11], 27, {0}, true, 0);
    for (int32_t i = 0; i < length; ++i) {
        llama_batch_add(batch.value, b_prompt[i], i, {1}, i + 1 == length, 1);
    }

    // Build metadata before profiling so JSON construction cannot enter t0..t2.
    json inputs = json::array();
    inputs.push_back({{"batch_index", 0}, {"token", a_generated[11]},
            {"position", 27}, {"sequence", 0}, {"adapter_id", 0}, {"output", true}});
    for (int32_t i = 0; i < length; ++i) {
        inputs.push_back({{"batch_index", i + 1}, {"token", b_prompt[i]},
                {"position", i}, {"sequence", 1}, {"adapter_id", 1},
                {"output", i + 1 == length}});
    }

    if (measured) {
        range.begin(capture && first_measurement, nvtx, "LCP/MEASURE/" + tag);
    }
    const int64_t t0 = monotonic_ns();
    const int rc = llama_decode(ctx, batch.value);
    const int64_t t1 = monotonic_ns();
    if (rc != 0) {
        throw std::runtime_error(tag + " llama_decode failed: " + std::to_string(rc));
    }
    llama_synchronize(ctx);
    const int64_t t2 = monotonic_ns();
    range.end(capture && last_measurement);

    // All output acquisition, validation, copies, serialization, and I/O are
    // outside the MEASURE NVTX range and synchronized-batch timing interval.
    // One profiler capture covers all measurements, so between-run work can
    // appear in the capture; filter by the explicit MEASURE ranges.
    const int32_t n_vocab = llama_n_vocab(model);
    const int32_t n_embd = llama_n_embd(model);
    const auto logits_a = copy_finite(llama_get_logits_ith(ctx, 0), n_vocab, tag + "/A logits");
    const auto logits_b = copy_finite(llama_get_logits_ith(ctx, length), n_vocab, tag + "/B logits");
    const auto hidden_a = copy_finite(llama_get_hidden_ith(ctx, 0), n_embd, tag + "/A hidden");
    const auto hidden_b = copy_finite(llama_get_hidden_ith(ctx, length), n_embd, tag + "/B hidden");
    const int64_t top_a = std::max_element(logits_a.begin(), logits_a.end()) - logits_a.begin();
    const int64_t top_b = std::max_element(logits_b.begin(), logits_b.end()) - logits_b.begin();

    json files = json::object();
    if (measured) {
        files = {{"A_logits", tag + "-A-logits.f32"}, {"B_logits", tag + "-B-logits.f32"},
                 {"A_hidden", tag + "-A-hidden.f32"}, {"B_hidden", tag + "-B-hidden.f32"}};
        dump_f32(root + "/" + files["A_logits"].get<std::string>(), logits_a);
        dump_f32(root + "/" + files["B_logits"].get<std::string>(), logits_b);
        dump_f32(root + "/" + files["A_hidden"].get<std::string>(), hidden_a);
        dump_f32(root + "/" + files["B_hidden"].get<std::string>(), hidden_b);
    }

    return {{"tag", tag}, {"measured", measured}, {"B_input_tokens", length},
            {"batch_n_tokens", length + 1}, {"batch_inputs", inputs},
            {"clock", "CLOCK_MONOTONIC"}, {"t0_ns", t0}, {"t1_ns", t1}, {"t2_ns", t2},
            {"decode_ms", (t1 - t0) / 1e6}, {"sync_ms", (t2 - t1) / 1e6},
            {"total_ms", (t2 - t0) / 1e6}, {"decode_rc", rc},
            {"topA", top_a}, {"topB", top_b},
            {"output_batch_indices", {0, length}}, {"output_rows", 2},
            {"logits_shape", {2, n_vocab}}, {"hidden_shape", {2, n_embd}},
            {"file_layout", "one contiguous native little-endian float32 row per file"},
            {"files", files}, {"capture_enabled", measured && capture},
            {"nvtx_enabled", measured && nvtx}};
}

int main(int argc, char ** argv) {
    llama_context * ctx = nullptr;
    llama_model * model = nullptr;
    bool backend_initialized = false;
    bool acquired_a = false;
    bool acquired_b = false;
    int result = 0;
    try {
        const std::string root = required_env("LCP_OUT");
        require_directory(root);
        const std::string plan_argument = required_env("LCP_PLAN");
        json plan;
        const size_t first = plan_argument.find_first_not_of(" \t\r\n");
        if (first != std::string::npos && plan_argument[first] == '{') {
            plan = json::parse(plan_argument);
        } else {
            std::ifstream input(plan_argument);
            if (!input) {
                throw std::runtime_error("cannot open LCP_PLAN: " + plan_argument);
            }
            input >> plan;
        }
        if (!plan.is_object() || !plan.at("prompts").is_object()) {
            throw std::runtime_error("invalid LCP_PLAN object");
        }
        const auto order = plan.at("length_order").get<std::vector<int>>();
        if (order.size() != 3 || std::set<int>(order.begin(), order.end()) != std::set<int>({16, 128, 256})) {
            throw std::runtime_error("length_order must contain 16, 128, and 256 exactly once");
        }
        require_absent(root + "/results.json");
        for (int length : order) {
            const std::string prefix = root + "/B" + std::to_string(length);
            for (const char * suffix : {"-A-logits.f32", "-B-logits.f32", "-A-hidden.f32", "-B-hidden.f32"}) {
                require_absent(prefix + suffix);
            }
        }

        gpt_params params;
        params.n_ctx = 2048;
        params.n_parallel = 2;
        params.n_batch = 512;
        params.n_ubatch = 512;
        auto options = gpt_params_parser_init(params, LLAMA_EXAMPLE_SERVER);
        if (!gpt_params_parse(argc, argv, params, options)) {
            return 2;
        }
        if (params.n_ctx != 2048 || params.n_parallel != 2 ||
                params.n_batch != 512 || params.n_ubatch != 512) {
            throw std::runtime_error("probe requires ctx=2048, parallel=2, batch=512, ubatch=512");
        }
        llama_backend_init();
        backend_initialized = true;
        auto init = llama_init_from_gpt_params(params);
        ctx = init.context;
        model = init.model;
        if (ctx == nullptr || model == nullptr) {
            throw std::runtime_error("model/context initialization failed");
        }
        if (llama_n_ctx(ctx) != 2048 || llama_n_batch(ctx) != 512 ||
                llama_n_ubatch(ctx) != 512 || llama_n_seq_max(ctx) < 2) {
            throw std::runtime_error("actual context dimensions do not match the experiment");
        }
        const int32_t n_vocab = llama_n_vocab(model);
        const auto a_prompt = get_tokens(plan.at("prompts").at("16"), 16, n_vocab, "A prompt");
        const auto a_generated = get_tokens(plan.at("a_generated"), 12, n_vocab, "A teacher tokens");
        for (int length : order) {
            get_tokens(plan.at("prompts").at(std::to_string(length)), length, n_vocab, "B prompt");
        }
        if (llama_lazy_load_lora_adapter(ctx, 0) != 0) {
            throw std::runtime_error("cannot acquire task adapter 0");
        }
        acquired_a = true;
        if (llama_lazy_load_lora_adapter(ctx, 1) != 0) {
            throw std::runtime_error("cannot acquire task adapter 1");
        }
        acquired_b = true;
        llama_synchronize(ctx);

        const bool capture = env_flag("LCP_CAPTURE");
        const bool nvtx = env_flag("LCP_NVTX");
        json output = {{"schema", "lora-cost-direct-batch-v1"},
                {"metric", "direct synchronized mixed batch time; not client TTFT"},
                {"capture_scope", "one start before first measured decode, one stop after last measured sync; filter MEASURE NVTX ranges"},
                {"length_order", order}, {"plan", plan},
                {"config", {{"n_ctx", llama_n_ctx(ctx)}, {"n_parallel", params.n_parallel},
                    {"n_batch", llama_n_batch(ctx)}, {"n_ubatch", llama_n_ubatch(ctx)},
                    {"n_seq_max", llama_n_seq_max(ctx)}, {"task_adapters", {0, 1}},
                    {"n_vocab", n_vocab}, {"n_embd", llama_n_embd(model)}}},
                {"preparation", {{"kv_clear_each_run", true}, {"A_sequence", 0},
                    {"A_adapter_id", 0}, {"A_prompt_tokens", a_prompt},
                    {"A_prompt_positions", {0, 15}}, {"A_prompt_outputs", "last token only"},
                    {"teacher_tokens_before_measurement", std::vector<llama_token>(a_generated.begin(), a_generated.begin() + 11)},
                    {"teacher_positions_before_measurement", {16, 26}},
                    {"teacher_step_batch_size", 1}, {"teacher_step_output", true},
                    {"all_preparation_decodes_synchronized", true},
                    {"measured_A_teacher_token", a_generated[11]}, {"measured_A_position", 27},
                    {"B_sequence", 1}, {"B_adapter_id", 1},
                    {"teacher_tokens_fixed_across_all_runs", true},
                    {"warmups_per_length", 1}, {"measurements_per_length", 1}}},
                {"warmups", json::array()}, {"measurements", json::array()}, {"complete", false}};

        CaptureRange range;
        for (int length : order) {
            const auto b_prompt = get_tokens(plan.at("prompts").at(std::to_string(length)),
                    length, n_vocab, "B prompt");
            output["warmups"].push_back(run_batch(ctx, model, a_prompt, a_generated,
                    b_prompt, root, false, false, false, range, false, false));
            std::printf("LCP warmup B%d complete\n", length);
            std::fflush(stdout);
        }
        for (size_t measurement = 0; measurement < order.size(); ++measurement) {
            const int length = order[measurement];
            const auto b_prompt = get_tokens(plan.at("prompts").at(std::to_string(length)),
                    length, n_vocab, "B prompt");
            output["measurements"].push_back(run_batch(ctx, model, a_prompt, a_generated,
                    b_prompt, root, true, capture, nvtx, range,
                    measurement == 0, measurement + 1 == order.size()));
            dump_json(root + "/results.json", output);
            const auto & row = output["measurements"].back();
            std::printf("LCP measured B%d decode_ms=%.6f sync_ms=%.6f total_ms=%.6f topA=%lld topB=%lld\n",
                    length, row["decode_ms"].get<double>(), row["sync_ms"].get<double>(),
                    row["total_ms"].get<double>(), static_cast<long long>(row["topA"].get<int64_t>()),
                    static_cast<long long>(row["topB"].get<int64_t>()));
            std::fflush(stdout);
        }
        output["complete"] = true;
        dump_json(root + "/results.json", output);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "LCP ERROR: %s\n", error.what());
        result = 1;
    } catch (...) {
        std::fprintf(stderr, "LCP ERROR: unknown exception\n");
        result = 1;
    }

    if (ctx != nullptr) {
        llama_synchronize(ctx);
        if (acquired_b) {
            llama_lora_adapter_release(ctx, 1);
        }
        if (acquired_a) {
            llama_lora_adapter_release(ctx, 0);
        }
        llama_free(ctx);
    }
    if (model != nullptr) {
        llama_free_model(model);
    }
    if (backend_initialized) {
        llama_backend_free();
    }
    return result;
}
