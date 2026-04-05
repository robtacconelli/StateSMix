#include "ssm_tokenizer.h"
#include "ssm_vocab.h"

/* ── Merge hash table ── */
#define MERGE_HT_CAP (1<<17)   /* 131072, ~2.6x load for 48900 entries */

typedef struct {
    uint32_t l, r, m;
    int32_t  rank;
    int      used;
} MergeHTE;

__attribute__((aligned(32))) static MergeHTE g_merge_ht[MERGE_HT_CAP];

static uint32_t merge_hash(uint32_t l, uint32_t r) {
    uint64_t k = ((uint64_t)l << 32) | r;
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33;
    return (uint32_t)(k & (MERGE_HT_CAP - 1));
}

static void merge_insert(uint32_t l, uint32_t r, uint32_t m, int32_t rank) {
    uint32_t h = merge_hash(l, r);
    while (g_merge_ht[h].used) h = (h+1) & (MERGE_HT_CAP-1);
    g_merge_ht[h] = (MergeHTE){l, r, m, rank, 1};
}

/* Returns rank >= 0 if found, -1 otherwise */
static int32_t merge_lookup(uint32_t l, uint32_t r, uint32_t *m_out) {
    uint32_t h = merge_hash(l, r);
    while (g_merge_ht[h].used) {
        if (g_merge_ht[h].l == l && g_merge_ht[h].r == r) {
            if (m_out) *m_out = g_merge_ht[h].m;
            return g_merge_ht[h].rank;
        }
        h = (h+1) & (MERGE_HT_CAP-1);
    }
    return -1;
}

/* ── Byte <-> token tables ── */
static int32_t  g_byte2tok[256];

#define TOK_BYTES_MAX (512*1024)
__attribute__((aligned(32))) static uint8_t  g_tok_data[TOK_BYTES_MAX];
static uint32_t g_tok_off[VOCAB+1];

/* ── Priority queue for BPE ── */
typedef struct { int32_t rank, pos, l, r; } PQE;

static void pq_sift_up(PQE *h, int i) {
    while (i > 0) {
        int p = (i-1)/2;
        if (h[p].rank <= h[i].rank) break;
        PQE tmp = h[p]; h[p] = h[i]; h[i] = tmp; i = p;
    }
}

static void pq_sift_dn(PQE *h, int n, int i) {
    for (;;) {
        int c = 2*i+1;
        if (c >= n) break;
        if (c+1 < n && h[c+1].rank < h[c].rank) c++;
        if (h[i].rank <= h[c].rank) break;
        PQE tmp = h[i]; h[i] = h[c]; h[c] = tmp; i = c;
    }
}

static void pq_push(PQE **h, int *n, int *cap, PQE e) {
    if (*n >= *cap) { *cap *= 2; *h = realloc(*h, *cap * sizeof(PQE)); }
    (*h)[(*n)++] = e; pq_sift_up(*h, *n - 1);
}

static PQE pq_pop(PQE *h, int *n) {
    PQE top = h[0]; h[0] = h[--(*n)];
    if (*n > 0) pq_sift_dn(h, *n, 0);
    return top;
}

/* ── Public functions ── */

int bpe_encode(const uint8_t *bytes, int n, int32_t **out) {
    if (n == 0) { *out = NULL; return 0; }

    int32_t *toks = malloc(n * sizeof(int32_t));
    for (int i = 0; i < n; i++) {
        int32_t t = g_byte2tok[bytes[i]];
        toks[i] = (t >= 0) ? t : 0;
    }

    int32_t *nxt = malloc(n * sizeof(int32_t));
    int32_t *prv = malloc(n * sizeof(int32_t));
    for (int i = 0; i < n; i++) { nxt[i] = i+1; prv[i] = i-1; }

    int pq_cap = n * 2 + 16, pq_n = 0;
    PQE *heap = malloc(pq_cap * sizeof(PQE));

    for (int i = 0; i < n-1; i++) {
        uint32_t m; int32_t rk = merge_lookup(toks[i], toks[i+1], &m);
        if (rk >= 0) pq_push(&heap, &pq_n, &pq_cap, (PQE){rk, i, toks[i], toks[i+1]});
    }

    while (pq_n > 0) {
        PQE e = pq_pop(heap, &pq_n);
        int32_t j = nxt[e.pos];
        if (j >= n) continue;
        if (toks[e.pos] != e.l || toks[j] != e.r) continue;

        uint32_t m; merge_lookup(toks[e.pos], toks[j], &m);
        toks[e.pos] = (int32_t)m;
        nxt[e.pos] = nxt[j];
        if (nxt[j] < n) prv[nxt[j]] = e.pos;

        if (prv[e.pos] >= 0) {
            int32_t p = prv[e.pos]; uint32_t m2;
            int32_t rk = merge_lookup(toks[p], toks[e.pos], &m2);
            if (rk >= 0) pq_push(&heap, &pq_n, &pq_cap,
                                  (PQE){rk, p, toks[p], toks[e.pos]});
        }
        if (nxt[e.pos] < n) {
            int32_t q = nxt[e.pos]; uint32_t m2;
            int32_t rk = merge_lookup(toks[e.pos], toks[q], &m2);
            if (rk >= 0) pq_push(&heap, &pq_n, &pq_cap,
                                  (PQE){rk, e.pos, toks[e.pos], toks[q]});
        }
    }

    int new_n = 0;
    for (int32_t i = 0; i < n; i = nxt[i]) {
        toks[new_n++] = toks[i];
        if (nxt[i] >= n) break;
    }

    free(nxt); free(prv); free(heap);
    *out = toks;
    return new_n;
}

int detokenize(const int32_t *toks, int n_toks, uint8_t *out) {
    int pos = 0;
    for (int i = 0; i < n_toks; i++) {
        int32_t cid = toks[i];
        if (cid < 0 || cid >= g_vocab_eff) continue;
        int32_t id = (int32_t)g_imap[cid];
        if (id < 0 || id >= VOCAB) continue;
        int len = (int)(g_tok_off[id+1] - g_tok_off[id]);
        memcpy(out + pos, g_tok_data + g_tok_off[id], len);
        pos += len;
    }
    return pos;
}

int load_tokenizer(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }

    char magic[4]; fread(magic, 1, 4, f);
    if (memcmp(magic, "BPET", 4) != 0) {
        fprintf(stderr, "Bad tokenizer magic\n"); fclose(f); return 1;
    }
    uint32_t vs, nm; fread(&vs, 4, 1, f); fread(&nm, 4, 1, f);
    if ((int)vs != VOCAB) {
        fprintf(stderr, "Vocab mismatch: file=%u code=%d\n", vs, VOCAB);
        fclose(f); return 1;
    }
    fread(g_byte2tok, 4, 256, f);

    uint32_t off = 0;
    for (int i = 0; i < VOCAB; i++) {
        g_tok_off[i] = off;
        uint8_t len; fread(&len, 1, 1, f);
        if (off + len < TOK_BYTES_MAX) {
            fread(g_tok_data + off, 1, len, f);
            off += len;
        } else {
            fseek(f, len, SEEK_CUR);
        }
    }
    g_tok_off[VOCAB] = off;

    memset(g_merge_ht, 0, sizeof(g_merge_ht));
    for (uint32_t i = 0; i < nm; i++) {
        uint32_t l, r, m;
        fread(&l, 4, 1, f); fread(&r, 4, 1, f); fread(&m, 4, 1, f);
        if (l < (uint32_t)VOCAB && r < (uint32_t)VOCAB && m < (uint32_t)VOCAB)
            merge_insert(l, r, m, (int32_t)i);
    }
    fclose(f);
    return 0;
}
