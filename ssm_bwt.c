#include "ssm_bwt.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Suffix array construction via prefix doubling ──
 * O(N log^2 N) time, O(N) space.
 * Works on circular suffixes for BWT. */

typedef struct { int idx; int r1; int r2; } SAPair;

static int sapair_cmp(const void *a, const void *b) {
    const SAPair *x = (const SAPair *)a;
    const SAPair *y = (const SAPair *)b;
    if (x->r1 != y->r1) return x->r1 - y->r1;
    if (x->r2 != y->r2) return x->r2 - y->r2;
    return 0;
}

static void build_suffix_array(const int32_t *toks, int n, int *sa) {
    /* Rank-based prefix doubling (Karp-Miller-Rosenberg style) */
    int *rank = malloc(n * sizeof(int));
    int *tmp  = malloc(n * sizeof(int));
    SAPair *pairs = malloc(n * sizeof(SAPair));

    /* Initial rank: by token value.
     * Map token values to dense ranks 0..k-1. */
    {
        int *sorted_idx = malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) sorted_idx[i] = i;
        /* Simple approach: assign rank = token value (works for comparison) */
        for (int i = 0; i < n; i++) rank[i] = toks[i];
        free(sorted_idx);
    }

    for (int h = 1; h < n; h *= 2) {
        /* Build pairs (rank[i], rank[(i+h) % n]) */
        for (int i = 0; i < n; i++) {
            pairs[i].idx = i;
            pairs[i].r1  = rank[i];
            pairs[i].r2  = rank[(i + h) % n];
        }
        qsort(pairs, n, sizeof(SAPair), sapair_cmp);

        /* Assign new ranks */
        tmp[pairs[0].idx] = 0;
        for (int i = 1; i < n; i++) {
            tmp[pairs[i].idx] = tmp[pairs[i-1].idx];
            if (pairs[i].r1 != pairs[i-1].r1 || pairs[i].r2 != pairs[i-1].r2)
                tmp[pairs[i].idx]++;
        }
        memcpy(rank, tmp, n * sizeof(int));

        /* If all ranks are unique, we're done */
        if (rank[pairs[n-1].idx] == n - 1) break;
    }

    /* Extract SA from final sorted order */
    for (int i = 0; i < n; i++) sa[rank[i]] = i;

    free(rank);
    free(tmp);
    free(pairs);
}

void bwt_forward(const int32_t *toks, int n, int32_t *out, uint32_t *primary_idx) {
    if (n <= 0) return;
    if (n == 1) { out[0] = toks[0]; *primary_idx = 0; return; }

    int *sa = malloc(n * sizeof(int));
    build_suffix_array(toks, n, sa);

    for (int i = 0; i < n; i++) {
        out[i] = toks[(sa[i] + n - 1) % n];
        if (sa[i] == 0) *primary_idx = (uint32_t)i;
    }

    free(sa);
}

void bwt_inverse(const int32_t *bwt, int n, int32_t *out, uint32_t primary_idx) {
    if (n <= 0) return;
    if (n == 1) { out[0] = bwt[0]; return; }

    /* Find max token value for array sizing */
    int32_t max_val = 0;
    for (int i = 0; i < n; i++)
        if (bwt[i] > max_val) max_val = bwt[i];

    int alpha = max_val + 1;

    /* C[c] = number of tokens in bwt[] that are strictly less than c */
    int *C = calloc(alpha + 1, sizeof(int));
    for (int i = 0; i < n; i++) C[bwt[i] + 1]++;
    for (int i = 1; i <= alpha; i++) C[i] += C[i - 1];

    /* Build LF mapping:
     * LF[i] = C[bwt[i]] + Occ(bwt[i], i)
     * where Occ(c, i) = number of occurrences of c in bwt[0..i-1] */
    int *LF  = malloc(n * sizeof(int));
    int *occ = calloc(alpha, sizeof(int));
    for (int i = 0; i < n; i++) {
        LF[i] = C[bwt[i]] + occ[bwt[i]];
        occ[bwt[i]]++;
    }

    /* Reconstruct original sequence in reverse order */
    int pos = (int)primary_idx;
    for (int i = n - 1; i >= 0; i--) {
        out[i] = bwt[pos];
        pos = LF[pos];
    }

    free(C);
    free(LF);
    free(occ);
}
