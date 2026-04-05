#include "ssm_preprocess.h"
#include "ssm_bwt.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int g_pp_mode = PP_NONE;
ExtraMerge g_extra_merges[MAX_EXTRA_MERGES];
int g_n_extra_merges = 0;

/* ── Reverse ── */
void reverse_toks(int32_t *toks, int n) {
    for (int i = 0, j = n - 1; i < j; i++, j--) {
        int32_t tmp = toks[i]; toks[i] = toks[j]; toks[j] = tmp;
    }
}

/* ── Extra BPE merges ──
 * Greedy: repeatedly find the most frequent adjacent pair and merge it.
 * Uses a hash table to count pair frequencies efficiently. */

#define PAIR_HT_SIZE (1 << 18)  /* 256K entries */
#define PAIR_HT_MASK (PAIR_HT_SIZE - 1)

typedef struct {
    int64_t key;     /* (left << 32) | right */
    int32_t count;
    int32_t valid;
} PairEntry;

static uint32_t pair_hash(int32_t l, int32_t r) {
    uint64_t k = ((uint64_t)(uint32_t)l << 32) | (uint32_t)r;
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33;
    return (uint32_t)(k & PAIR_HT_MASK);
}

int extra_bpe_apply(int32_t *toks, int n_toks, int min_freq, int max_merges) {
    if (max_merges > MAX_EXTRA_MERGES) max_merges = MAX_EXTRA_MERGES;
    g_n_extra_merges = 0;

    /* Use linked list for efficient gap handling (same as BPE encoder) */
    int32_t *nxt = malloc(n_toks * sizeof(int32_t));
    int32_t *prv = malloc(n_toks * sizeof(int32_t));
    for (int i = 0; i < n_toks; i++) { nxt[i] = i + 1; prv[i] = i - 1; }

    PairEntry *ht = calloc(PAIR_HT_SIZE, sizeof(PairEntry));
    int new_id = EXTRA_BPE_BASE;

    for (int round = 0; round < max_merges; round++) {
        /* Count all adjacent pairs */
        memset(ht, 0, PAIR_HT_SIZE * sizeof(PairEntry));

        int best_count = 0;
        int64_t best_key = -1;

        for (int i = 0; i < n_toks; ) {
            int j = nxt[i];
            if (j >= n_toks) { i = j; continue; }

            int64_t key = ((int64_t)(uint32_t)toks[i] << 32) | (uint32_t)toks[j];
            uint32_t h = pair_hash(toks[i], toks[j]);
            while (ht[h].valid && ht[h].key != key) h = (h + 1) & PAIR_HT_MASK;

            if (!ht[h].valid) {
                ht[h].key = key;
                ht[h].count = 1;
                ht[h].valid = 1;
            } else {
                ht[h].count++;
            }

            if (ht[h].count > best_count) {
                best_count = ht[h].count;
                best_key = key;
            }

            i = j;
        }

        if (best_count < min_freq) break;

        /* Record this merge */
        int32_t left  = (int32_t)(uint32_t)(best_key >> 32);
        int32_t right = (int32_t)(uint32_t)(best_key & 0xFFFFFFFF);
        g_extra_merges[g_n_extra_merges].left = left;
        g_extra_merges[g_n_extra_merges].right = right;
        g_n_extra_merges++;

        /* Apply merge: replace all (left, right) → new_id */
        int merged = 0;
        for (int i = 0; i < n_toks; ) {
            int j = nxt[i];
            if (j >= n_toks) { i = j; continue; }

            if (toks[i] == left && toks[j] == right) {
                toks[i] = new_id;
                /* Remove position j from linked list */
                nxt[i] = nxt[j];
                if (nxt[j] < n_toks) prv[nxt[j]] = i;
                merged++;
                /* Don't advance - check if new pair at i can merge again
                 * (it can't with same merge, but for consistency) */
                i = nxt[i];
            } else {
                i = j;
            }
        }
        new_id++;
    }

    /* Compact the array */
    int new_n = 0;
    for (int i = 0; i < n_toks; i = nxt[i]) {
        toks[new_n++] = toks[i];
        if (nxt[i] >= n_toks) break;
    }

    free(nxt);
    free(prv);
    free(ht);

    return new_n;
}

int extra_bpe_expand(const int32_t *toks, int n_toks, int32_t *out) {
    /* First pass: expand super-tokens into original BPE tokens.
     * Super-tokens have IDs >= EXTRA_BPE_BASE.
     * Must expand in reverse merge order (last merge first). */

    /* Copy input to working buffer */
    int cap = n_toks * 4;  /* worst case: each token expands to 2^n_merges tokens */
    int32_t *work = malloc(cap * sizeof(int32_t));
    memcpy(work, toks, n_toks * sizeof(int32_t));
    int n = n_toks;

    /* Expand merges in reverse order */
    for (int m = g_n_extra_merges - 1; m >= 0; m--) {
        int32_t mid = EXTRA_BPE_BASE + m;
        int32_t left  = g_extra_merges[m].left;
        int32_t right = g_extra_merges[m].right;

        /* Count how many expansions we'll do */
        int expand_count = 0;
        for (int i = 0; i < n; i++)
            if (work[i] == mid) expand_count++;

        if (expand_count == 0) continue;

        /* Expand in-place (right to left to avoid overwriting) */
        int new_n = n + expand_count;
        if (new_n > cap) {
            cap = new_n * 2;
            work = realloc(work, cap * sizeof(int32_t));
        }

        /* Copy from right to left */
        int dst = new_n - 1;
        for (int src = n - 1; src >= 0; src--) {
            if (work[src] == mid) {
                work[dst--] = right;
                work[dst--] = left;
            } else {
                work[dst--] = work[src];
            }
        }
        n = new_n;
    }

    memcpy(out, work, n * sizeof(int32_t));
    free(work);
    return n;
}
