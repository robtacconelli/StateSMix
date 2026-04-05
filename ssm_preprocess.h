#ifndef SSM_PREPROCESS_H
#define SSM_PREPROCESS_H

#include <stdint.h>

/* ── Preprocessing modes ── */
#define PP_NONE    0
#define PP_REVERSE 1
#define PP_BWT     2
#define PP_EXTBPE  3

/* Global preprocessing mode */
extern int g_pp_mode;

/* ── Extra BPE merges ── */
typedef struct {
    int32_t left;   /* original (or previously merged) token ID */
    int32_t right;
} ExtraMerge;

#define EXTRA_BPE_BASE  49152   /* new token IDs start here */
#define MAX_EXTRA_MERGES 4096

extern ExtraMerge g_extra_merges[MAX_EXTRA_MERGES];
extern int g_n_extra_merges;

/* Apply extra BPE merges to a token sequence (original IDs).
 * min_freq: minimum pair frequency to merge.
 * max_merges: maximum number of merges.
 * Modifies toks[] in-place, returns new token count. */
int extra_bpe_apply(int32_t *toks, int n_toks, int min_freq, int max_merges);

/* Expand extra BPE super-tokens back to original BPE tokens.
 * out must have enough space (up to 2x n_toks in worst case).
 * Returns number of output tokens. */
int extra_bpe_expand(const int32_t *toks, int n_toks, int32_t *out);

/* ── Reverse ── */
void reverse_toks(int32_t *toks, int n);

/* ── BWT (re-export from ssm_bwt.h) ── */
void bwt_forward(const int32_t *toks, int n, int32_t *out, uint32_t *primary_idx);
void bwt_inverse(const int32_t *bwt, int n, int32_t *out, uint32_t primary_idx);

#endif
