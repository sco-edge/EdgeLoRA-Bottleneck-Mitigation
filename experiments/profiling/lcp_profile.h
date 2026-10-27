#pragma once
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <nvToolsExt.h>
#include "ggml.h"

static inline bool lcp_enabled() {
    static const bool enabled = std::getenv("LCP_NVTX") && std::strcmp(std::getenv("LCP_NVTX"), "1") == 0;
    return enabled;
}
struct lcp_scope {
    bool active;
    explicit lcp_scope(const char *name) : active(lcp_enabled()) { if (active) nvtxRangePushA(name); }
    ~lcp_scope() { if (active) nvtxRangePop(); }
};
static inline bool lcp_weight(const ggml_tensor *t, const char *suffix) {
    return t && std::strstr(t->name, suffix) != nullptr;
}
static inline const char *lcp_kind(const ggml_tensor *t) {
    if (!t) return "OTHER";
    if (t->op == GGML_OP_MUL_MAT && lcp_weight(t->src[0], ".lora_a")) return "LORA_A";
    if (t->op == GGML_OP_MUL_MAT && lcp_weight(t->src[0], ".lora_b")) return "LORA_B";
    if (t->op == GGML_OP_SCALE && t->src[0] && t->src[0]->op == GGML_OP_MUL_MAT && lcp_weight(t->src[0]->src[0], ".lora_b")) return "LORA_SCALE";
    if (t->op == GGML_OP_ACC && t->src[1] && std::strcmp(lcp_kind(t->src[1]), "LORA_SCALE") == 0) return "LORA_ACC";
    return "OTHER";
}
struct lcp_node_scope {
    bool active;
    explicit lcp_node_scope(const ggml_tensor *t) : active(lcp_enabled()) {
        if (active) {
            char label[384];
            std::snprintf(label, sizeof(label), "LCP/NODE/%s|%s|%s|s0=%s|bytes=%zu|ne1=%lld|alias=%d",
                lcp_kind(t), ggml_op_name(t->op), t->name, t->src[0] ? t->src[0]->name : "",
                ggml_nbytes(t), (long long)t->ne[1], t->src[0] && t->data == t->src[0]->data);
            nvtxRangePushA(label);
        }
    }
    ~lcp_node_scope() { if (active) nvtxRangePop(); }
};
