#include "ssm_vocab.h"

int      g_vocab_eff = VOCAB;
int32_t  g_remap[REMAP_MAX];
uint32_t g_imap[VOCAB];

void build_vocab_map(int32_t *toks, int n) {
    memset(g_remap, -1, sizeof(g_remap));
    g_vocab_eff = 0;
    for (int i = 0; i < n; i++) {
        int32_t t = toks[i];
        if (t < 0 || t >= REMAP_MAX) continue;
        if (g_remap[t] < 0) {
            g_remap[t] = g_vocab_eff;
            g_imap[g_vocab_eff] = (uint32_t)t;
            g_vocab_eff++;
        }
    }
}

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t*)a, y = *(const uint32_t*)b;
    return (x > y) - (x < y);
}

void sort_vocab_map(void) {
    qsort(g_imap, g_vocab_eff, sizeof(uint32_t), cmp_u32);
    memset(g_remap, -1, REMAP_MAX * sizeof(int32_t));
    for (int i = 0; i < g_vocab_eff; i++)
        g_remap[g_imap[i]] = (int32_t)i;
}

void remap_toks(int32_t *toks, int n) {
    for (int i = 0; i < n; i++) toks[i] = g_remap[toks[i]];
}
