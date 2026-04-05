#ifndef SSM_BWT_H
#define SSM_BWT_H

#include <stdint.h>

/* Token-level Burrows-Wheeler Transform.
 * Operates on compact token IDs (int32_t), not bytes.
 * Overhead: 4 bytes (BWT primary index). */

/* Forward BWT: toks[0..n-1] → out[0..n-1] + primary index.
 * out must be pre-allocated to n int32_t. */
void bwt_forward(const int32_t *toks, int n, int32_t *out, uint32_t *primary_idx);

/* Inverse BWT: bwt[0..n-1] + primary index → out[0..n-1].
 * out must be pre-allocated to n int32_t. */
void bwt_inverse(const int32_t *bwt, int n, int32_t *out, uint32_t primary_idx);

#endif
