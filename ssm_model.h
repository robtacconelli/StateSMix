#ifndef SSM_MODEL_H
#define SSM_MODEL_H

#include "ssm_config.h"

/* ── Layer parameters ── */
typedef struct __attribute__((aligned(32))) {
    float ln_g[DM], ln_b[DM];
    float in_w[DM * 2*DI];
    float conv_w[DI * DC], conv_b[DI];
    float xp_w[DI * XP_DIM];
    float dt_w[DI], dt_b[DI];
    float A_log[DI * DS];
    float D[DI];
    float out_w[DI * DM];
} LParams;

/* Full model parameters.
 * head_w is token-major [VOCAB x DM] so the first g_vocab_eff*DM floats
 * are contiguous, matching the emb layout. */
typedef struct __attribute__((aligned(32))) {
    LParams L[NL];
    float fn_g[DM], fn_b[DM];
    float emb[VOCAB * DM];
    float head_w[VOCAB * DM];
} Params;

/* Offset of first vocab float in Params (LParams x NL + fn_g + fn_b) */
#define FIXED_FLOATS (19776)

/* ── SSM state ── */
typedef struct __attribute__((aligned(32))) {
    float h[NL][DI * DS];
    float cb[NL][DI * (DC-1)];
} State;

/* ── Per-layer forward cache (for backprop) ── */
typedef struct {
    float x_pre[DM], x_hat[DM];
    float inv_std, x_norm[DM];
    float xs[DI], z[DI];
    float full[DI * DC];
    float xc_pre[DI], xc[DI], sig_xc[DI];
    float B[DS], C[DS];
    float dt_in, dt_pre[DI], dt[DI];
    float h_old[DI * DS], h_new[DI * DS];
    float dA_cache[DI * DS];
    float y[DI];
    float sig_z[DI], gate[DI], out_pre[DI];
} LCache;

/* ── Full forward cache ── */
typedef struct __attribute__((aligned(32))) {
    int bv;
    LCache lc[NL];
    float x_pfn[DM], x_hat_fn[DM];
    float inv_std_fn, x_final[DM];
} SCache;

/* ── Precomputed -exp(A_log) ── */
extern float g_A_neg[NL][DI * DS];

/* ── Shared temp buffers ── */
extern float g_tc_logits[VOCAB];
extern float g_tc_probs[VOCAB];
extern float g_tc_dlogits[VOCAB];
extern int32_t g_cdf[VOCAB+1];
extern int    g_tok_count[VOCAB];
extern float  g_mixed_lg[VOCAB];

/* ── Adaptive learning rate ── */
extern float g_lr;

/* ── Functions ── */
void recompute_A_neg(const Params *p);
void init_params(Params *p);
void forward_token(const Params * __restrict__ p, State * __restrict__ st, int bv,
                   float * __restrict__ logits, SCache * __restrict__ c);
void backward_token(const Params * __restrict__ p, const SCache * __restrict__ c,
                    const float * __restrict__ dlogits, Params * __restrict__ g);
void do_softmax(const float * __restrict__ logits, float * __restrict__ probs);
int32_t probs_to_cdf(const float * __restrict__ probs);
void adam_update(Params * __restrict__ params, Params * __restrict__ grads,
                 Params * __restrict__ am, Params * __restrict__ av, int n, int step);
void train_chunk(Params * __restrict__ params, Params * __restrict__ am,
                 Params * __restrict__ av, Params * __restrict__ grads,
                 const int32_t * __restrict__ chunk, int clen,
                 const State * __restrict__ st0, int * __restrict__ astep,
                 State * __restrict__ st_out, float * __restrict__ last_lg_out,
                 int train_iters);

#endif /* SSM_MODEL_H */
