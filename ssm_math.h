#ifndef SSM_MATH_H
#define SSM_MATH_H

#include "ssm_config.h"

/* ── PRNG ── */
extern uint64_t g_rng;
float randn(void);

/* ── Fast math (static inline for performance) ── */

static inline float fast_expf_a(float x) {
    if (x > 87.0f) return 3.4e38f;
    if (x < -87.0f) return 0.0f;
    float t = x * 1.4426950408889634f;
    int n = (int)(t + (t >= 0.0f ? 0.5f : -0.5f));
    float r = x - (float)n * 0.6931471805599453f;
    float p = 1.0f + r*(1.0f + r*(0.5f + r*(0.16666667f + r*0.04166667f)));
    union { float f; int32_t i; } u;
    u.i = (n + 127) << 23;
    return p * u.f;
}

/* Branchless exp for softmax hot loop — clamps silently, vectorises with AVX2. */
static inline float fast_expf_v(float x) {
    x = x < -87.0f ? -87.0f : (x > 87.0f ? 87.0f : x);
    float n_f = roundf(x * 1.4426950408889634f);
    float r   = x - n_f * 0.6931471805599453f;
    float p   = 1.0f + r*(1.0f + r*(0.5f + r*(0.16666667f + r*0.04166667f)));
    union { float f; int32_t i; } u;
    u.i = ((int)n_f + 127) << 23;
    return p * u.f;
}

static inline float fast_logf_a(float x) {
    if (x <= 0.0f) return -87.0f;
    union { float f; uint32_t u; } v = {x};
    int e = (int)(v.u >> 23) - 127;
    v.u = (v.u & 0x007fffffU) | 0x3f800000U;
    float m = v.f;
    float t2_ = (m - 1.0f) / (m + 1.0f);
    float t2  = t2_ * t2_;
    float log_m = 2.0f * t2_ * (1.0f + t2*(0.33333333f + t2*(0.2f + t2*0.14285714f)));
    return log_m + (float)e * 0.6931471805599453f;
}

static inline float fast_sigmoid(float x) {
    if (x > 30.0f) return 1.0f;
    if (x < -30.0f) return 0.0f;
    return 1.0f / (1.0f + fast_expf_a(-x));
}

static inline float fast_softplus(float x) {
    if (x > 30.0f) return x;
    if (x < -30.0f) return 0.0f;
    return fast_logf_a(1.0f + fast_expf_a(x));
}

#endif /* SSM_MATH_H */
