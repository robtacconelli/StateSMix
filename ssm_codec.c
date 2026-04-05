#define _POSIX_C_SOURCE 200112L
#include "ssm_codec.h"
#include "ssm_config.h"
#include "ssm_math.h"
#include "ssm_vocab.h"
#include "ssm_tokenizer.h"
#include "ssm_arithmetic.h"
#include "ssm_model.h"
#include "ssm_preprocess.h"
#include <limits.h>
#include <string.h>

/* ── Rice coder ── */
typedef struct { uint8_t *buf; int nbytes, nbits; } BitW;
static void bw_init(BitW *b, uint8_t *buf) { b->buf=buf; b->nbytes=0; b->nbits=0; }
static void bw_bit(BitW *b, int bit) {
    if (!b->nbits) b->buf[b->nbytes] = 0;
    b->buf[b->nbytes] |= (uint8_t)((bit&1) << (7-b->nbits));
    if (++b->nbits == 8) { b->nbytes++; b->nbits = 0; }
}
static void bw_flush(BitW *b) { if (b->nbits) { b->nbytes++; b->nbits=0; } }
static void bw_rice(BitW *b, uint32_t n, int k) {
    uint32_t q = n >> k, r = n & ((1u<<k)-1);
    for (uint32_t i=0; i<q; i++) bw_bit(b,0);
    bw_bit(b,1);
    for (int i=k-1; i>=0; i--) bw_bit(b,(r>>i)&1);
}
typedef struct { const uint8_t *buf; int nbytes, nbits; } BitR;
static void br_init(BitR *b, const uint8_t *buf) { b->buf=buf; b->nbytes=0; b->nbits=0; }
static int br_bit(BitR *b) {
    int bit = (b->buf[b->nbytes] >> (7-b->nbits)) & 1;
    if (++b->nbits == 8) { b->nbytes++; b->nbits=0; }
    return bit;
}
static uint32_t br_rice(BitR *b, int k) {
    uint32_t q=0, r=0;
    while (!br_bit(b)) q++;
    for (int i=k-1; i>=0; i--) r |= (uint32_t)br_bit(b)<<i;
    return (q<<k)|r;
}

#define ALIGN32_SZ(sz) (((sz) + 31) & ~(size_t)31)

/* ══════════════════════════════════════════════════════
 * Sparse n-gram models with softmax-invariant logit bias.
 *
 * Key insight: softmax(x + c) = softmax(x), so the constant
 * background bias lambda * log(alpha/denom) applied to ALL
 * tokens can be dropped. Only non-zero counts contribute:
 *
 *   logits[j] += lambda * log(1 + count[j] / alpha)
 *
 * This makes sparse storage both memory- and compute-efficient.
 * ══════════════════════════════════════════════════════ */

/* ── Sparse count structure (per-context) ── */
typedef struct {
    uint16_t *toks;     /* token IDs (compact) */
    uint16_t *cnts;     /* counts (capped at 65535) */
    int       n;        /* number of entries */
    int       cap;      /* allocated capacity */
    int       total;    /* sum of all counts */
} SparseCtx;

static void sc_update(SparseCtx *s, int tok) {
    /* Linear search — fine for typical fan-out (5-30 entries) */
    uint16_t t = (uint16_t)tok;
    for (int i = 0; i < s->n; i++) {
        if (s->toks[i] == t) {
            if (s->cnts[i] < 65535) s->cnts[i]++;
            s->total++;
            return;
        }
    }
    /* New entry */
    if (s->n >= s->cap) {
        int nc = s->cap ? s->cap * 2 : 4;
        s->toks = realloc(s->toks, nc * sizeof(uint16_t));
        s->cnts = realloc(s->cnts, nc * sizeof(uint16_t));
        s->cap = nc;
    }
    s->toks[s->n] = t;
    s->cnts[s->n] = 1;
    s->n++;
    s->total++;
}

static void sc_free(SparseCtx *s) {
    free(s->toks); free(s->cnts);
    s->toks = NULL; s->cnts = NULL;
    s->n = s->cap = 0; s->total = 0;
}

/* Decay all counts by right-shifting. Removes entries that hit zero. */
static void sc_decay(SparseCtx *s) {
    int w = 0;
    int new_total = 0;
    for (int i = 0; i < s->n; i++) {
        uint16_t c = s->cnts[i] >> 1;
        if (c > 0) {
            s->toks[w] = s->toks[i];
            s->cnts[w] = c;
            new_total += c;
            w++;
        }
    }
    s->n = w;
    s->total = new_total;
}

/* Softmax-invariant logit bias: only touches non-zero entries.
 * Adds lambda * log(1 + count/alpha) for each stored token. */
static void sc_add_logits(const SparseCtx *s, float *logits, float lambda, float alpha) {
    if (s->n == 0) return;
    float inv_alpha = 1.0f / alpha;
    for (int i = 0; i < s->n; i++) {
        float c = (float)s->cnts[i];
        logits[s->toks[i]] += lambda * logf(1.0f + c * inv_alpha);
    }
}

/* ── Bigram model (sparse) ── */
#define BIGRAM_LAMBDA 0.15f
#define BIGRAM_ALPHA 0.1f

static SparseCtx *o1_ctx;   /* array of ve sparse contexts */

static void bigram_init(int ve) {
    o1_ctx = calloc(ve, sizeof(SparseCtx));
}
static void bigram_free(int ve) {
    for (int i = 0; i < ve; i++) sc_free(&o1_ctx[i]);
    free(o1_ctx); o1_ctx = NULL;
}
static void bigram_decay(int ve) {
    for (int i = 0; i < ve; i++) sc_decay(&o1_ctx[i]);
}
static void bigram_update(int ctx, int tok, int ve) {
    (void)ve;
    sc_update(&o1_ctx[ctx], tok);
}
static void bigram_add_logits_scaled(int ctx, float *logits, int ve, float scale) {
    (void)ve;
    if (ctx < 0) return;
    sc_add_logits(&o1_ctx[ctx], logits, BIGRAM_LAMBDA * scale, BIGRAM_ALPHA);
}

/* ── LZ hash predictor (logit-space) ── */
#define LZ_HASH_SIZE (1<<24)
#define LZ_HASH_MASK (LZ_HASH_SIZE-1)
#define LZ_BOOST 1.5f

typedef struct {
    uint32_t key;
    int32_t tok;
    int16_t count;
    int16_t valid;
} LZEntry;

static LZEntry *lz_table;

static void lz_init(void) {
    lz_table = calloc(LZ_HASH_SIZE, sizeof(LZEntry));
}
static void lz_free(void) { free(lz_table); lz_table = NULL; }

static uint32_t lz_hash(int prev2, int prev1) {
    return (uint32_t)(prev2 * 7919 + prev1 * 104729 + 1000003) & LZ_HASH_MASK;
}

static void lz_update(int prev2, int prev1, int tok) {
    uint32_t h = lz_hash(prev2, prev1);
    uint32_t full_key = ((uint32_t)(uint16_t)prev2 << 16) | (uint32_t)(uint16_t)prev1;
    LZEntry *e = &lz_table[h];
    if (e->valid && e->key == full_key && e->tok == tok) {
        if (e->count < 30000) e->count++;
    } else if (!e->valid) {
        e->key = full_key; e->tok = tok; e->count = 1; e->valid = 1;
    } else if (e->key != full_key) {
        /* Different context hashed to same slot — evict if weak */
        if (e->count <= 1) {
            e->key = full_key; e->tok = tok; e->count = 1;
        } else {
            e->count--;
        }
    } else {
        /* Same context, different token — evict if weak */
        if (e->count <= 1) {
            e->tok = tok; e->count = 1;
        } else {
            e->count--;
        }
    }
}

static void lz_add_logits_scaled(int prev2, int prev1, float *logits, float scale) {
    if (prev2 < 0 || prev1 < 0) return;
    uint32_t h = lz_hash(prev2, prev1);
    uint32_t full_key = ((uint32_t)(uint16_t)prev2 << 16) | (uint32_t)(uint16_t)prev1;
    LZEntry *e = &lz_table[h];
    if (e->valid && e->key == full_key) {
        float conf = (float)e->count;
        float boost = (LZ_BOOST * scale) * (1.0f - 1.0f/(1.0f + conf*0.3f));
        logits[e->tok] += boost;
    }
}

/* ── Recency bias ── */
#define RECENCY_WINDOW 64
#define RECENCY_LAMBDA 0.05f

static int32_t recency_buf[RECENCY_WINDOW];
static int recency_pos = 0, recency_count = 0;

static void recency_init(void) { recency_pos = 0; recency_count = 0; }
static void recency_add(int tok) {
    recency_buf[recency_pos % RECENCY_WINDOW] = tok;
    recency_pos++;
    if (recency_count < RECENCY_WINDOW) recency_count++;
}
static void recency_add_logits_scaled(float *logits, float scale) {
    if (recency_count == 0) return;
    float lambda = RECENCY_LAMBDA * scale;
    for (int i = 0; i < recency_count; i++) {
        int start = recency_pos - recency_count;
        int tok = recency_buf[(start + i) % RECENCY_WINDOW];
        float age = (float)(recency_count - 1 - i) / RECENCY_WINDOW;
        float bonus = lambda * expf(-3.0f * age);
        logits[tok] += bonus;
    }
}

/* ── Generic sparse hash-table n-gram ── */
typedef struct {
    uint64_t   key;    /* full context key (not hash!) for proper collision rejection */
    SparseCtx  sc;
    int        valid;
} SparseNgramSlot;

/* Linear probing depth — try up to PROBE_LEN slots before giving up */
#define PROBE_LEN 8

/* Tokens seen so far — used for adaptive lambda scaling */
static int g_tokens_seen = 0;

/* Configurable hash table size: override with -DNGRAM_HASH_LOG2=26 etc. */
#ifndef NGRAM_HASH_LOG2
#define NGRAM_HASH_LOG2 24
#endif
#define NGRAM_HASH_SIZE  (1u << NGRAM_HASH_LOG2)
#define NGRAM_HASH_MASK  (NGRAM_HASH_SIZE - 1u)

/* ── Dynamic resize support (compile with -DEXP_DYNRESIZE) ─────────── */
#ifdef EXP_DYNRESIZE
typedef uint32_t (*ngram_rehash_fn)(uint64_t key, uint32_t mask);

typedef struct {
    SparseNgramSlot *slots;
    uint32_t size;
    uint32_t mask;
    uint32_t count;
    ngram_rehash_fn rehash;
} DynTable;

static void dyn_table_init(DynTable *t, uint32_t initial_size, ngram_rehash_fn rh) {
    t->size = initial_size;
    t->mask = initial_size - 1;
    t->count = 0;
    t->rehash = rh;
    t->slots = calloc(initial_size, sizeof(SparseNgramSlot));
}

static void dyn_table_free(DynTable *t) {
    for (uint32_t i = 0; i < t->size; i++) sc_free(&t->slots[i].sc);
    free(t->slots); t->slots = NULL;
    t->size = t->mask = t->count = 0;
}

static void dyn_table_grow(DynTable *t) {
    uint32_t new_size = t->size * 2;
    uint32_t new_mask = new_size - 1;
    SparseNgramSlot *new_slots = calloc(new_size, sizeof(SparseNgramSlot));
    for (uint32_t i = 0; i < t->size; i++) {
        if (t->slots[i].valid) {
            uint32_t h = t->rehash(t->slots[i].key, new_mask);
            for (int p = 0; p < PROBE_LEN * 4; p++) {
                SparseNgramSlot *e = &new_slots[(h + (uint32_t)p) & new_mask];
                if (!e->valid) { *e = t->slots[i]; break; }
            }
        }
    }
    free(t->slots);  /* sc buffers moved, not freed */
    t->slots = new_slots;
    t->size = new_size;
    t->mask = new_mask;
}

static void dyn_table_decay(DynTable *t) {
    for (uint32_t i = 0; i < t->size; i++)
        if (t->slots[i].valid) sc_decay(&t->slots[i].sc);
}

static inline void dyn_slot_update(DynTable *t, uint32_t h, uint64_t fk, int tok) {
    for (int p = 0; p < PROBE_LEN; p++) {
        SparseNgramSlot *e = &t->slots[(h + (uint32_t)p) & t->mask];
        if (!e->valid) {
            e->key = fk; e->valid = 1; sc_update(&e->sc, tok);
            t->count++;
            if (t->count * 4 > t->size * 3)  /* > 75% load */
                dyn_table_grow(t);
            return;
        }
        if (e->key == fk) { sc_update(&e->sc, tok); return; }
    }
}

static inline void dyn_slot_add_logits(DynTable *t, uint32_t h,
                                        uint64_t fk, float *logits, float lambda, float alpha) {
    for (int p = 0; p < PROBE_LEN; p++) {
        SparseNgramSlot *e = &t->slots[(h + (uint32_t)p) & t->mask];
        if (!e->valid) return;
        if (e->key == fk) { sc_add_logits(&e->sc, logits, lambda, alpha); return; }
    }
}

/* Per-order rehash functions (extract tokens from packed keys) */
static uint32_t tri_rehash(uint64_t key, uint32_t mask) {
    uint32_t p1 = (uint32_t)(key & 0xFFFF), p2 = (uint32_t)((key >> 16) & 0xFFFF);
    return (p2 * 104729u + p1 * 7919u) & mask;
}
static uint32_t four_rehash(uint64_t key, uint32_t mask) {
    uint32_t p1 = (uint32_t)(key & 0xFFFF), p2 = (uint32_t)((key >> 16) & 0xFFFF),
             p3 = (uint32_t)((key >> 32) & 0xFFFF);
    return (p3 * 1000003u + p2 * 104729u + p1 * 7919u) & mask;
}
static uint32_t five_rehash(uint64_t key, uint32_t mask) {
    uint32_t p1 = (uint32_t)(key & 0xFFFF), p2 = (uint32_t)((key >> 16) & 0xFFFF),
             p3 = (uint32_t)((key >> 32) & 0xFFFF), p4 = (uint32_t)((key >> 48) & 0xFFFF);
    return (p4 * 16777259u + p3 * 1000003u + p2 * 104729u + p1 * 7919u) & mask;
}
static uint32_t highorder_rehash(uint64_t key, uint32_t mask) {
    return (uint32_t)key & mask;  /* key is already mix64(polynomial) */
}

/* DynTable instances for each order */
static DynTable dyn_tri, dyn_four, dyn_five, dyn_six, dyn_seven, dyn_eight;
#endif /* EXP_DYNRESIZE */

/* Generic probing helpers shared by all n-gram orders */
static inline void ngram_slot_update(SparseNgramSlot *table, uint32_t h, uint32_t mask,
                                     uint64_t fk, int tok) {
    for (int p = 0; p < PROBE_LEN; p++) {
        SparseNgramSlot *e = &table[(h + (uint32_t)p) & mask];
        if (!e->valid) { e->key = fk; e->valid = 1; sc_update(&e->sc, tok); return; }
        if (e->key == fk) { sc_update(&e->sc, tok); return; }
    }
#ifdef EXP_EVICT
    /* LFU eviction: replace the probe-window slot with lowest total count */
    {
        int min_total = INT_MAX, min_idx = 0;
        for (int p = 0; p < PROBE_LEN; p++) {
            SparseNgramSlot *e = &table[(h + (uint32_t)p) & mask];
            if (e->sc.total < min_total) { min_total = e->sc.total; min_idx = p; }
        }
        SparseNgramSlot *victim = &table[(h + (uint32_t)min_idx) & mask];
        sc_free(&victim->sc);
        memset(&victim->sc, 0, sizeof(SparseCtx));
        victim->key = fk;
        sc_update(&victim->sc, tok);
    }
#endif
}

static inline void ngram_slot_add_logits(SparseNgramSlot *table, uint32_t h, uint32_t mask,
                                          uint64_t fk, float *logits, float lambda, float alpha) {
    for (int p = 0; p < PROBE_LEN; p++) {
        SparseNgramSlot *e = &table[(h + (uint32_t)p) & mask];
        if (!e->valid) return;
        if (e->key == fk) { sc_add_logits(&e->sc, logits, lambda, alpha); return; }
    }
}

/* Trigram: order-2 (prev2, prev1) → token counts */
#define TRI_HASH_SIZE  NGRAM_HASH_SIZE
#define TRI_HASH_MASK  NGRAM_HASH_MASK
#define TRIGRAM_LAMBDA  0.10f
#define TRIGRAM_ALPHA   0.05f

static inline uint32_t tri_hash_raw(int p2, int p1) {
    return (uint32_t)p2 * 104729u + (uint32_t)p1 * 7919u;
}
static inline uint64_t tri_key(int p2, int p1) {
    return ((uint64_t)(uint16_t)p2 << 16) | (uint64_t)(uint16_t)p1;
}

#ifndef EXP_DYNRESIZE
static SparseNgramSlot *tri_table;
static void trigram_init(void) { tri_table = calloc(TRI_HASH_SIZE, sizeof(SparseNgramSlot)); }
static void trigram_free(void) {
    for (int i = 0; i < TRI_HASH_SIZE; i++) sc_free(&tri_table[i].sc);
    free(tri_table); tri_table = NULL;
}
static void trigram_decay(void) {
    for (int i = 0; i < TRI_HASH_SIZE; i++)
        if (tri_table[i].valid) sc_decay(&tri_table[i].sc);
}
static void trigram_update(int p2, int p1, int tok, int ve) {
    if (p2 < 0 || p1 < 0) return; (void)ve;
    uint32_t h = tri_hash_raw(p2, p1) & TRI_HASH_MASK;
    ngram_slot_update(tri_table, h, TRI_HASH_MASK, tri_key(p2,p1), tok);
}
static void trigram_add_logits_scaled(int p2, int p1, float *logits, int ve, float scale) {
    if (p2 < 0 || p1 < 0) return; (void)ve;
    uint32_t h = tri_hash_raw(p2, p1) & TRI_HASH_MASK;
    ngram_slot_add_logits(tri_table, h, TRI_HASH_MASK, tri_key(p2,p1), logits, TRIGRAM_LAMBDA*scale, TRIGRAM_ALPHA);
}
#else /* EXP_DYNRESIZE */
static void trigram_init(void) { dyn_table_init(&dyn_tri, NGRAM_HASH_SIZE, tri_rehash); }
static void trigram_free(void) { dyn_table_free(&dyn_tri); }
static void trigram_decay(void) { dyn_table_decay(&dyn_tri); }
static void trigram_update(int p2, int p1, int tok, int ve) {
    if (p2 < 0 || p1 < 0) return; (void)ve;
    dyn_slot_update(&dyn_tri, tri_hash_raw(p2,p1), tri_key(p2,p1), tok);
}
static void trigram_add_logits_scaled(int p2, int p1, float *logits, int ve, float scale) {
    if (p2 < 0 || p1 < 0) return; (void)ve;
    dyn_slot_add_logits(&dyn_tri, tri_hash_raw(p2,p1), tri_key(p2,p1), logits, TRIGRAM_LAMBDA*scale, TRIGRAM_ALPHA);
}
#endif

/* 4-gram: order-3 (prev3, prev2, prev1) */
#define FOUR_HASH_SIZE  NGRAM_HASH_SIZE
#define FOUR_HASH_MASK  NGRAM_HASH_MASK
#define FOURGRAM_LAMBDA 0.08f
#define FOURGRAM_ALPHA  0.03f

static inline uint32_t four_hash_raw(int p3, int p2, int p1) {
    return (uint32_t)p3 * 1000003u + (uint32_t)p2 * 104729u + (uint32_t)p1 * 7919u;
}
static inline uint64_t four_key(int p3, int p2, int p1) {
    return ((uint64_t)(uint16_t)p3 << 32) | ((uint64_t)(uint16_t)p2 << 16) | (uint64_t)(uint16_t)p1;
}

#ifndef EXP_DYNRESIZE
static SparseNgramSlot *four_table;
static void fourgram_init(void) { four_table = calloc(FOUR_HASH_SIZE, sizeof(SparseNgramSlot)); }
static void fourgram_free(void) {
    for (int i = 0; i < FOUR_HASH_SIZE; i++) sc_free(&four_table[i].sc);
    free(four_table); four_table = NULL;
}
static void fourgram_decay(void) {
    for (int i = 0; i < FOUR_HASH_SIZE; i++)
        if (four_table[i].valid) sc_decay(&four_table[i].sc);
}
static void fourgram_update(int p3, int p2, int p1, int tok, int ve) {
    if (p3 < 0 || p2 < 0 || p1 < 0) return; (void)ve;
    uint32_t h = four_hash_raw(p3,p2,p1) & FOUR_HASH_MASK;
    ngram_slot_update(four_table, h, FOUR_HASH_MASK, four_key(p3,p2,p1), tok);
}
static void fourgram_add_logits_scaled(int p3, int p2, int p1, float *logits, int ve, float scale) {
    if (p3 < 0 || p2 < 0 || p1 < 0) return; (void)ve;
    uint32_t h = four_hash_raw(p3,p2,p1) & FOUR_HASH_MASK;
    ngram_slot_add_logits(four_table, h, FOUR_HASH_MASK, four_key(p3,p2,p1), logits, FOURGRAM_LAMBDA*scale, FOURGRAM_ALPHA);
}
#else /* EXP_DYNRESIZE */
static void fourgram_init(void) { dyn_table_init(&dyn_four, NGRAM_HASH_SIZE, four_rehash); }
static void fourgram_free(void) { dyn_table_free(&dyn_four); }
static void fourgram_decay(void) { dyn_table_decay(&dyn_four); }
static void fourgram_update(int p3, int p2, int p1, int tok, int ve) {
    if (p3 < 0 || p2 < 0 || p1 < 0) return; (void)ve;
    dyn_slot_update(&dyn_four, four_hash_raw(p3,p2,p1), four_key(p3,p2,p1), tok);
}
static void fourgram_add_logits_scaled(int p3, int p2, int p1, float *logits, int ve, float scale) {
    if (p3 < 0 || p2 < 0 || p1 < 0) return; (void)ve;
    dyn_slot_add_logits(&dyn_four, four_hash_raw(p3,p2,p1), four_key(p3,p2,p1), logits, FOURGRAM_LAMBDA*scale, FOURGRAM_ALPHA);
}
#endif

/* 5-gram: order-4 (prev4, prev3, prev2, prev1) */
#define FIVE_HASH_SIZE  NGRAM_HASH_SIZE
#define FIVE_HASH_MASK  NGRAM_HASH_MASK
#define FIVEGRAM_LAMBDA 0.06f
#define FIVEGRAM_ALPHA  0.02f

static inline uint32_t five_hash_raw(int p4, int p3, int p2, int p1) {
    return (uint32_t)p4 * 16777259u + (uint32_t)p3 * 1000003u +
           (uint32_t)p2 * 104729u + (uint32_t)p1 * 7919u;
}
static inline uint64_t five_key(int p4, int p3, int p2, int p1) {
    return ((uint64_t)(uint16_t)p4 << 48) | ((uint64_t)(uint16_t)p3 << 32) |
           ((uint64_t)(uint16_t)p2 << 16) | (uint64_t)(uint16_t)p1;
}

#ifndef EXP_DYNRESIZE
static SparseNgramSlot *five_table;
static void fivegram_init(void) { five_table = calloc(FIVE_HASH_SIZE, sizeof(SparseNgramSlot)); }
static void fivegram_free(void) {
    for (int i = 0; i < FIVE_HASH_SIZE; i++) sc_free(&five_table[i].sc);
    free(five_table); five_table = NULL;
}
static void fivegram_decay(void) {
    for (int i = 0; i < FIVE_HASH_SIZE; i++)
        if (five_table[i].valid) sc_decay(&five_table[i].sc);
}
static void fivegram_update(int p4, int p3, int p2, int p1, int tok, int ve) {
    if (p4 < 0 || p3 < 0 || p2 < 0 || p1 < 0) return; (void)ve;
    uint32_t h = five_hash_raw(p4,p3,p2,p1) & FIVE_HASH_MASK;
    ngram_slot_update(five_table, h, FIVE_HASH_MASK, five_key(p4,p3,p2,p1), tok);
}
static void fivegram_add_logits_scaled(int p4, int p3, int p2, int p1, float *logits, int ve, float scale) {
    if (p4 < 0 || p3 < 0 || p2 < 0 || p1 < 0) return; (void)ve;
    uint32_t h = five_hash_raw(p4,p3,p2,p1) & FIVE_HASH_MASK;
    ngram_slot_add_logits(five_table, h, FIVE_HASH_MASK, five_key(p4,p3,p2,p1), logits, FIVEGRAM_LAMBDA*scale, FIVEGRAM_ALPHA);
}
#else /* EXP_DYNRESIZE */
static void fivegram_init(void) { dyn_table_init(&dyn_five, NGRAM_HASH_SIZE, five_rehash); }
static void fivegram_free(void) { dyn_table_free(&dyn_five); }
static void fivegram_decay(void) { dyn_table_decay(&dyn_five); }
static void fivegram_update(int p4, int p3, int p2, int p1, int tok, int ve) {
    if (p4 < 0 || p3 < 0 || p2 < 0 || p1 < 0) return; (void)ve;
    dyn_slot_update(&dyn_five, five_hash_raw(p4,p3,p2,p1), five_key(p4,p3,p2,p1), tok);
}
static void fivegram_add_logits_scaled(int p4, int p3, int p2, int p1, float *logits, int ve, float scale) {
    if (p4 < 0 || p3 < 0 || p2 < 0 || p1 < 0) return; (void)ve;
    dyn_slot_add_logits(&dyn_five, five_hash_raw(p4,p3,p2,p1), five_key(p4,p3,p2,p1), logits, FIVEGRAM_LAMBDA*scale, FIVEGRAM_ALPHA);
}
#endif

/* 6-gram: order-5 */
#define SIX_HASH_SIZE   NGRAM_HASH_SIZE
#define SIX_HASH_MASK   NGRAM_HASH_MASK
#define SIXGRAM_LAMBDA  0.05f
#define SIXGRAM_ALPHA   0.015f

static inline uint64_t mix64(uint64_t h) {
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33; return h;
}

static uint64_t six_key(int p5, int p4, int p3, int p2, int p1) {
    uint64_t k = (uint64_t)(uint16_t)p5;
    k = k * 104729 + (uint64_t)(uint16_t)p4;
    k = k * 104729 + (uint64_t)(uint16_t)p3;
    k = k * 104729 + (uint64_t)(uint16_t)p2;
    k = k * 104729 + (uint64_t)(uint16_t)p1;
    return mix64(k);
}

#ifndef EXP_DYNRESIZE
static SparseNgramSlot *six_table;
static void sixgram_init(void) { six_table = calloc(SIX_HASH_SIZE, sizeof(SparseNgramSlot)); }
static void sixgram_free(void) {
    for (int i = 0; i < SIX_HASH_SIZE; i++) sc_free(&six_table[i].sc);
    free(six_table); six_table = NULL;
}
static void sixgram_decay(void) {
    for (int i = 0; i < SIX_HASH_SIZE; i++)
        if (six_table[i].valid) sc_decay(&six_table[i].sc);
}
static void sixgram_update(int p5, int p4, int p3, int p2, int p1, int tok) {
    if (p5 < 0 || p4 < 0 || p3 < 0 || p2 < 0 || p1 < 0) return;
    uint64_t fk = six_key(p5,p4,p3,p2,p1);
    ngram_slot_update(six_table, (uint32_t)fk & SIX_HASH_MASK, SIX_HASH_MASK, fk, tok);
}
static void sixgram_add_logits_scaled(int p5, int p4, int p3, int p2, int p1, float *logits, float scale) {
    if (p5 < 0 || p4 < 0 || p3 < 0 || p2 < 0 || p1 < 0) return;
    uint64_t fk = six_key(p5,p4,p3,p2,p1);
    ngram_slot_add_logits(six_table, (uint32_t)fk & SIX_HASH_MASK, SIX_HASH_MASK, fk, logits, SIXGRAM_LAMBDA*scale, SIXGRAM_ALPHA);
}
#else /* EXP_DYNRESIZE */
static void sixgram_init(void) { dyn_table_init(&dyn_six, NGRAM_HASH_SIZE, highorder_rehash); }
static void sixgram_free(void) { dyn_table_free(&dyn_six); }
static void sixgram_decay(void) { dyn_table_decay(&dyn_six); }
static void sixgram_update(int p5, int p4, int p3, int p2, int p1, int tok) {
    if (p5 < 0 || p4 < 0 || p3 < 0 || p2 < 0 || p1 < 0) return;
    uint64_t fk = six_key(p5,p4,p3,p2,p1);
    dyn_slot_update(&dyn_six, (uint32_t)fk, fk, tok);
}
static void sixgram_add_logits_scaled(int p5, int p4, int p3, int p2, int p1, float *logits, float scale) {
    if (p5 < 0 || p4 < 0 || p3 < 0 || p2 < 0 || p1 < 0) return;
    uint64_t fk = six_key(p5,p4,p3,p2,p1);
    dyn_slot_add_logits(&dyn_six, (uint32_t)fk, fk, logits, SIXGRAM_LAMBDA*scale, SIXGRAM_ALPHA);
}
#endif

/* 7-gram: order-6 */
#define SEVEN_HASH_SIZE  NGRAM_HASH_SIZE
#define SEVEN_HASH_MASK  NGRAM_HASH_MASK
#define SEVENGRAM_LAMBDA 0.04f
#define SEVENGRAM_ALPHA  0.01f

static uint64_t seven_key(int p6, int p5, int p4, int p3, int p2, int p1) {
    uint64_t k = (uint64_t)(uint16_t)p6;
    k = k * 104729 + (uint64_t)(uint16_t)p5;
    k = k * 104729 + (uint64_t)(uint16_t)p4;
    k = k * 104729 + (uint64_t)(uint16_t)p3;
    k = k * 104729 + (uint64_t)(uint16_t)p2;
    k = k * 104729 + (uint64_t)(uint16_t)p1;
    return mix64(k);
}

#ifndef EXP_DYNRESIZE
static SparseNgramSlot *seven_table;
static void sevengram_init(void) { seven_table = calloc(SEVEN_HASH_SIZE, sizeof(SparseNgramSlot)); }
static void sevengram_free(void) {
    for (int i = 0; i < SEVEN_HASH_SIZE; i++) sc_free(&seven_table[i].sc);
    free(seven_table); seven_table = NULL;
}
static void sevengram_decay(void) {
    for (int i = 0; i < SEVEN_HASH_SIZE; i++)
        if (seven_table[i].valid) sc_decay(&seven_table[i].sc);
}
static void sevengram_update(int p6, int p5, int p4, int p3, int p2, int p1, int tok) {
    if (p6 < 0 || p5 < 0 || p4 < 0 || p3 < 0 || p2 < 0 || p1 < 0) return;
    uint64_t fk = seven_key(p6,p5,p4,p3,p2,p1);
    ngram_slot_update(seven_table, (uint32_t)fk & SEVEN_HASH_MASK, SEVEN_HASH_MASK, fk, tok);
}
static void sevengram_add_logits_scaled(int p6, int p5, int p4, int p3, int p2, int p1, float *logits, float scale) {
    if (p6 < 0 || p5 < 0 || p4 < 0 || p3 < 0 || p2 < 0 || p1 < 0) return;
    uint64_t fk = seven_key(p6,p5,p4,p3,p2,p1);
    ngram_slot_add_logits(seven_table, (uint32_t)fk & SEVEN_HASH_MASK, SEVEN_HASH_MASK, fk, logits, SEVENGRAM_LAMBDA*scale, SEVENGRAM_ALPHA);
}
#else /* EXP_DYNRESIZE */
static void sevengram_init(void) { dyn_table_init(&dyn_seven, NGRAM_HASH_SIZE, highorder_rehash); }
static void sevengram_free(void) { dyn_table_free(&dyn_seven); }
static void sevengram_decay(void) { dyn_table_decay(&dyn_seven); }
static void sevengram_update(int p6, int p5, int p4, int p3, int p2, int p1, int tok) {
    if (p6 < 0 || p5 < 0 || p4 < 0 || p3 < 0 || p2 < 0 || p1 < 0) return;
    uint64_t fk = seven_key(p6,p5,p4,p3,p2,p1);
    dyn_slot_update(&dyn_seven, (uint32_t)fk, fk, tok);
}
static void sevengram_add_logits_scaled(int p6, int p5, int p4, int p3, int p2, int p1, float *logits, float scale) {
    if (p6 < 0 || p5 < 0 || p4 < 0 || p3 < 0 || p2 < 0 || p1 < 0) return;
    uint64_t fk = seven_key(p6,p5,p4,p3,p2,p1);
    dyn_slot_add_logits(&dyn_seven, (uint32_t)fk, fk, logits, SEVENGRAM_LAMBDA*scale, SEVENGRAM_ALPHA);
}
#endif

/* 8-gram: order-7 */
#define EIGHT_HASH_SIZE  NGRAM_HASH_SIZE
#define EIGHT_HASH_MASK  NGRAM_HASH_MASK
#define EIGHTGRAM_LAMBDA 0.03f
#define EIGHTGRAM_ALPHA  0.008f

static uint64_t eight_key(int p7, int p6, int p5, int p4, int p3, int p2, int p1) {
    uint64_t k = (uint64_t)(uint16_t)p7;
    k = k * 104729 + (uint64_t)(uint16_t)p6;
    k = k * 104729 + (uint64_t)(uint16_t)p5;
    k = k * 104729 + (uint64_t)(uint16_t)p4;
    k = k * 104729 + (uint64_t)(uint16_t)p3;
    k = k * 104729 + (uint64_t)(uint16_t)p2;
    k = k * 104729 + (uint64_t)(uint16_t)p1;
    return mix64(k);
}

#ifndef EXP_DYNRESIZE
static SparseNgramSlot *eight_table;
static void eightgram_init(void) { eight_table = calloc(EIGHT_HASH_SIZE, sizeof(SparseNgramSlot)); }
static void eightgram_free(void) {
    for (int i = 0; i < EIGHT_HASH_SIZE; i++) sc_free(&eight_table[i].sc);
    free(eight_table); eight_table = NULL;
}
static void eightgram_update(int p7, int p6, int p5, int p4, int p3, int p2, int p1, int tok) {
    if (p7 < 0 || p6 < 0 || p5 < 0 || p4 < 0 || p3 < 0 || p2 < 0 || p1 < 0) return;
    uint64_t fk = eight_key(p7,p6,p5,p4,p3,p2,p1);
    ngram_slot_update(eight_table, (uint32_t)fk & EIGHT_HASH_MASK, EIGHT_HASH_MASK, fk, tok);
}
static void eightgram_add_logits_scaled(int p7, int p6, int p5, int p4, int p3, int p2, int p1, float *logits, float scale) {
    if (p7 < 0 || p6 < 0 || p5 < 0 || p4 < 0 || p3 < 0 || p2 < 0 || p1 < 0) return;
    uint64_t fk = eight_key(p7,p6,p5,p4,p3,p2,p1);
    ngram_slot_add_logits(eight_table, (uint32_t)fk & EIGHT_HASH_MASK, EIGHT_HASH_MASK, fk, logits, EIGHTGRAM_LAMBDA*scale, EIGHTGRAM_ALPHA);
}
#else /* EXP_DYNRESIZE */
static void eightgram_init(void) { dyn_table_init(&dyn_eight, NGRAM_HASH_SIZE, highorder_rehash); }
static void eightgram_free(void) { dyn_table_free(&dyn_eight); }
static void eightgram_update(int p7, int p6, int p5, int p4, int p3, int p2, int p1, int tok) {
    if (p7 < 0 || p6 < 0 || p5 < 0 || p4 < 0 || p3 < 0 || p2 < 0 || p1 < 0) return;
    uint64_t fk = eight_key(p7,p6,p5,p4,p3,p2,p1);
    dyn_slot_update(&dyn_eight, (uint32_t)fk, fk, tok);
}
static void eightgram_add_logits_scaled(int p7, int p6, int p5, int p4, int p3, int p2, int p1, float *logits, float scale) {
    if (p7 < 0 || p6 < 0 || p5 < 0 || p4 < 0 || p3 < 0 || p2 < 0 || p1 < 0) return;
    uint64_t fk = eight_key(p7,p6,p5,p4,p3,p2,p1);
    dyn_slot_add_logits(&dyn_eight, (uint32_t)fk, fk, logits, EIGHTGRAM_LAMBDA*scale, EIGHTGRAM_ALPHA);
}
#endif

#ifndef EXP_DYNRESIZE
static void eightgram_decay(void) {
    for (int i = 0; i < EIGHT_HASH_SIZE; i++)
        if (eight_table[i].valid) sc_decay(&eight_table[i].sc);
}
#else
static void eightgram_decay(void) { dyn_table_decay(&dyn_eight); }
#endif

/* ── Long-range context matching: 16-gram and 32-gram ── */
#define SIXTEEN_HASH_SIZE  NGRAM_HASH_SIZE
#define SIXTEEN_HASH_MASK  NGRAM_HASH_MASK
#define SIXTEEN_LAMBDA     0.50f
#define SIXTEEN_ALPHA      0.001f
#define SIXTEEN_CTX        15       /* 15 tokens of context */

#define THIRTY2_HASH_SIZE  NGRAM_HASH_SIZE
#define THIRTY2_HASH_MASK  NGRAM_HASH_MASK
#define THIRTY2_LAMBDA     1.00f
#define THIRTY2_ALPHA      0.001f
#define THIRTY2_CTX        31       /* 31 tokens of context */

static uint64_t long_ctx_key(const int32_t *toks, int pos, int ctx_len) {
    uint64_t k = 0;
    for (int i = pos - ctx_len; i < pos; i++)
        k = k * 104729 + (uint64_t)(uint16_t)toks[i];
    return mix64(k);
}

#ifndef EXP_DYNRESIZE
static SparseNgramSlot *sixteen_table;
static void sixteengram_init(void) { sixteen_table = calloc(SIXTEEN_HASH_SIZE, sizeof(SparseNgramSlot)); }
static void sixteengram_free(void) {
    for (int i = 0; i < SIXTEEN_HASH_SIZE; i++) sc_free(&sixteen_table[i].sc);
    free(sixteen_table); sixteen_table = NULL;
}
static void sixteengram_update(const int32_t *toks, int pos, int tok) {
    if (pos < SIXTEEN_CTX) return;
    uint64_t fk = long_ctx_key(toks, pos, SIXTEEN_CTX);
    uint32_t h = (uint32_t)fk & SIXTEEN_HASH_MASK;
    ngram_slot_update(sixteen_table, h, SIXTEEN_HASH_MASK, fk, tok);
}
static void sixteengram_add_logits(const int32_t *toks, int pos, float *logits, float scale) {
    if (pos < SIXTEEN_CTX) return;
    uint64_t fk = long_ctx_key(toks, pos, SIXTEEN_CTX);
    uint32_t h = (uint32_t)fk & SIXTEEN_HASH_MASK;
    ngram_slot_add_logits(sixteen_table, h, SIXTEEN_HASH_MASK, fk, logits, SIXTEEN_LAMBDA * scale, SIXTEEN_ALPHA);
}

static SparseNgramSlot *thirty2_table;
static void thirty2gram_init(void) { thirty2_table = calloc(THIRTY2_HASH_SIZE, sizeof(SparseNgramSlot)); }
static void thirty2gram_free(void) {
    for (int i = 0; i < THIRTY2_HASH_SIZE; i++) sc_free(&thirty2_table[i].sc);
    free(thirty2_table); thirty2_table = NULL;
}
static void thirty2gram_update(const int32_t *toks, int pos, int tok) {
    if (pos < THIRTY2_CTX) return;
    uint64_t fk = long_ctx_key(toks, pos, THIRTY2_CTX);
    uint32_t h = (uint32_t)fk & THIRTY2_HASH_MASK;
    ngram_slot_update(thirty2_table, h, THIRTY2_HASH_MASK, fk, tok);
}
static void thirty2gram_add_logits(const int32_t *toks, int pos, float *logits, float scale) {
    if (pos < THIRTY2_CTX) return;
    uint64_t fk = long_ctx_key(toks, pos, THIRTY2_CTX);
    uint32_t h = (uint32_t)fk & THIRTY2_HASH_MASK;
    ngram_slot_add_logits(thirty2_table, h, THIRTY2_HASH_MASK, fk, logits, THIRTY2_LAMBDA * scale, THIRTY2_ALPHA);
}
#else /* EXP_DYNRESIZE */
static DynTable dyn_sixteen, dyn_thirty2;
static void sixteengram_init(void) { dyn_table_init(&dyn_sixteen, NGRAM_HASH_SIZE, highorder_rehash); }
static void sixteengram_free(void) { dyn_table_free(&dyn_sixteen); }
static void sixteengram_update(const int32_t *toks, int pos, int tok) {
    if (pos < SIXTEEN_CTX) return;
    uint64_t fk = long_ctx_key(toks, pos, SIXTEEN_CTX);
    dyn_slot_update(&dyn_sixteen, (uint32_t)fk, fk, tok);
}
static void sixteengram_add_logits(const int32_t *toks, int pos, float *logits, float scale) {
    if (pos < SIXTEEN_CTX) return;
    uint64_t fk = long_ctx_key(toks, pos, SIXTEEN_CTX);
    dyn_slot_add_logits(&dyn_sixteen, (uint32_t)fk, fk, logits, SIXTEEN_LAMBDA * scale, SIXTEEN_ALPHA);
}
static void thirty2gram_init(void) { dyn_table_init(&dyn_thirty2, NGRAM_HASH_SIZE, highorder_rehash); }
static void thirty2gram_free(void) { dyn_table_free(&dyn_thirty2); }
static void thirty2gram_update(const int32_t *toks, int pos, int tok) {
    if (pos < THIRTY2_CTX) return;
    uint64_t fk = long_ctx_key(toks, pos, THIRTY2_CTX);
    dyn_slot_update(&dyn_thirty2, (uint32_t)fk, fk, tok);
}
static void thirty2gram_add_logits(const int32_t *toks, int pos, float *logits, float scale) {
    if (pos < THIRTY2_CTX) return;
    uint64_t fk = long_ctx_key(toks, pos, THIRTY2_CTX);
    dyn_slot_add_logits(&dyn_thirty2, (uint32_t)fk, fk, logits, THIRTY2_LAMBDA * scale, THIRTY2_ALPHA);
}
#endif

/* ── LR decay & n-gram count decay ── */
#define LR_DECAY_RATE   0.0f        /* 0 = disabled */
#define NGRAM_DECAY_INTERVAL 0      /* 0 = disabled */

/* Ablation mode: 0=full, 1=SSM+count only, 2=ngram+count only, 3=count only */
static int g_ablation = 0;

/* ── Entropy-adaptive scaling ── */
#define ENTROPY_TARGET 5.5f
#define ENTROPY_MIN_SCALE 0.2f
#define ENTROPY_MAX_SCALE 2.5f
#define ENTROPY_BLEND 0.6f

static float compute_ssm_entropy(const float *logits, int ve) {
    float mx = logits[0];
    float sum = 0, H = 0;
    static float probs_tmp[VOCAB];
    #pragma omp parallel
    {
        #pragma omp for reduction(max:mx) schedule(static)
        for (int j = 1; j < ve; j++) if (logits[j] > mx) mx = logits[j];

        #pragma omp for reduction(+:sum) schedule(static)
        for (int j = 0; j < ve; j++) { probs_tmp[j] = expf(logits[j] - mx); sum += probs_tmp[j]; }

        float inv = 1.0f / sum;
        #pragma omp for reduction(+:H) schedule(static)
        for (int j = 0; j < ve; j++) {
            float p = probs_tmp[j] * inv;
            if (p > 1e-10f) H -= p * logf(p);
        }
    }
    return H;
}

/* ── Combined logit computation (entropy-adaptive) ── */
static void compute_combined_probs(int has_lg, int prev1, int prev2,
                                    int prev3, int prev4, int prev5,
                                    int prev6, int prev7,
                                    float *last_lg, float *p_out, int ve,
                                    const int32_t *toks, int pos) {
    /* ablation mode 2/3: treat as if SSM has never run (count prior only) */
    int eff_has_lg = (g_ablation == 2 || g_ablation == 3) ? 0 : has_lg;

    float count_scale;
    if (!eff_has_lg) {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < ve; j++)
            g_mixed_lg[j] = COUNT_LAMBDA * logf((float)(g_tok_count[j]+1));
        count_scale = 1.0f;
    } else {
        float H = compute_ssm_entropy(last_lg, ve);
        float ratio = H / ENTROPY_TARGET;
        count_scale = (1.0f - ENTROPY_BLEND) + ENTROPY_BLEND * ratio;
        if (count_scale < ENTROPY_MIN_SCALE) count_scale = ENTROPY_MIN_SCALE;
        if (count_scale > ENTROPY_MAX_SCALE) count_scale = ENTROPY_MAX_SCALE;
        float cl_cs = COUNT_LAMBDA * count_scale;
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < ve; j++)
            g_mixed_lg[j] = last_lg[j] + cl_cs * logf((float)(g_tok_count[j]+1));
    }
    /* ablation mode 1/3: skip all n-gram contributions */
    if (g_ablation == 1 || g_ablation == 3) { do_softmax(g_mixed_lg, p_out); return; }

    float ngram_scale = count_scale;
    bigram_add_logits_scaled(prev1, g_mixed_lg, ve, ngram_scale);
    trigram_add_logits_scaled(prev2, prev1, g_mixed_lg, ve, ngram_scale);
    fourgram_add_logits_scaled(prev3, prev2, prev1, g_mixed_lg, ve, ngram_scale);
    fivegram_add_logits_scaled(prev4, prev3, prev2, prev1, g_mixed_lg, ve, ngram_scale);
    sixgram_add_logits_scaled(prev5, prev4, prev3, prev2, prev1, g_mixed_lg, ngram_scale);
    sevengram_add_logits_scaled(prev6, prev5, prev4, prev3, prev2, prev1, g_mixed_lg, ngram_scale);
    eightgram_add_logits_scaled(prev7, prev6, prev5, prev4, prev3, prev2, prev1, g_mixed_lg, ngram_scale);
    sixteengram_add_logits(toks, pos, g_mixed_lg, ngram_scale);
    thirty2gram_add_logits(toks, pos, g_mixed_lg, ngram_scale);
    lz_add_logits_scaled(prev2, prev1, g_mixed_lg, ngram_scale);
    recency_add_logits_scaled(g_mixed_lg, ngram_scale);
    do_softmax(g_mixed_lg, p_out);
}

/* ── Compress ── */
int do_compress(const char *inpath, const char *outpath) {
    { const char *ab = getenv("ABLATION"); if (ab) g_ablation = atoi(ab); }
    static const char *ablation_names[] = {"full","ssm+count","ngram+count","count_only"};
    if (g_ablation) printf("[ABLATION=%d: %s]\n", g_ablation, ablation_names[g_ablation]);

    FILE *f=fopen(inpath,"rb");
    if(!f){perror(inpath);return 1;}
    fseek(f,0,SEEK_END); long fsize=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t *raw=malloc(fsize);
    fread(raw,1,fsize,f); fclose(f);
    printf("Input: %s (%ld bytes)\n",inpath,fsize);

    struct timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    int32_t *toks; int n_toks = bpe_encode(raw, (int)fsize, &toks);
    free(raw);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double tok_time = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    printf("Tokenized: %d tokens (%.3f tok/byte) in %.2fs\n",
           n_toks, (double)n_toks/fsize, tok_time);

    /* ── Preprocessing (on original token IDs, before vocab mapping) ── */
    uint32_t bwt_idx = 0;
    size_t pp_overhead = 0;
    if (g_pp_mode == PP_EXTBPE) {
        int orig_n = n_toks;
        int ebpe_minfreq = 50, ebpe_maxmerge = 100;
        { /* Check env for tuning */
            const char *mf = getenv("EBPE_MINFREQ");
            const char *mm = getenv("EBPE_MAXMERGE");
            if (mf) ebpe_minfreq = atoi(mf);
            if (mm) ebpe_maxmerge = atoi(mm);
        }
        n_toks = extra_bpe_apply(toks, n_toks, ebpe_minfreq, ebpe_maxmerge);
        pp_overhead = 2 + (size_t)g_n_extra_merges * 8; /* uint16 count + 8B per merge */
        printf("Extra BPE: %d merges, %d -> %d tokens (%.1f%% reduction), overhead=%zuB\n",
               g_n_extra_merges, orig_n, n_toks,
               100.0*(1.0-(double)n_toks/orig_n), pp_overhead);
    }

    build_vocab_map(toks, n_toks);
    sort_vocab_map();
    remap_toks(toks, n_toks);
    printf("Compact vocab: %d unique tokens (head: %d×%d = %dk ops/tok vs %dk)\n",
           g_vocab_eff, DM, g_vocab_eff, DM*g_vocab_eff/1000, DM*VOCAB/1000);

    /* ── Post-remap preprocessing (on compact IDs) ── */
    if (g_pp_mode == PP_REVERSE) {
        reverse_toks(toks, n_toks);
        pp_overhead = 0;
        printf("Reversed token sequence\n");
    } else if (g_pp_mode == PP_BWT) {
        int32_t *bwt_buf = malloc(n_toks * sizeof(int32_t));
        bwt_forward(toks, n_toks, bwt_buf, &bwt_idx);
        memcpy(toks, bwt_buf, n_toks * sizeof(int32_t));
        free(bwt_buf);
        pp_overhead = 4;
        printf("Applied token BWT (index=%u)\n", bwt_idx);
    }

    size_t params_sz = ALIGN32_SZ(sizeof(Params));
    size_t state_sz  = ALIGN32_SZ(sizeof(State));
    size_t scache_sz = ALIGN32_SZ(sizeof(SCache));
    Params *params = aligned_alloc(32, params_sz); memset(params, 0, params_sz);
    Params *am     = aligned_alloc(32, params_sz); memset(am,     0, params_sz);
    Params *av     = aligned_alloc(32, params_sz); memset(av,     0, params_sz);
    Params *grads  = aligned_alloc(32, params_sz); memset(grads,  0, params_sz);
    init_params(params);
    recompute_A_neg(params);

    State  *state    = aligned_alloc(32, state_sz);  memset(state,    0, state_sz);
    State  *chunk_st = aligned_alloc(32, state_sz);  memset(chunk_st, 0, state_sz);
    SCache *cache    = aligned_alloc(32, scache_sz); memset(cache,    0, scache_sz);
    AEnc enc; aenc_init(&enc);

    uint64_t N_orig = (uint64_t)fsize;
    uint64_t N_toks = (uint64_t)n_toks;
    (void)bwt_idx; /* used in header for BWT mode */

    int32_t cbuf[CHUNK_SZ];
    int clen=0, astep=0, has_lg=0, chunk_num=0;
    float *last_lg=calloc(g_vocab_eff,sizeof(float));
    double total_bits=0;

    int ve = g_vocab_eff;
    bigram_init(ve);
    lz_init();
    trigram_init();
    fourgram_init();
    fivegram_init();
    sixgram_init();
    sevengram_init();
    eightgram_init();
    sixteengram_init();
    thirty2gram_init();
    recency_init();
    g_lr = ADAM_LR;
    g_tokens_seen = 0;
    int prev1=-1, prev2=-1, prev3=-1, prev4=-1, prev5=-1, prev6=-1, prev7=-1;
    int next_decay = NGRAM_DECAY_INTERVAL > 0 ? NGRAM_DECAY_INTERVAL : INT32_MAX;

    clock_gettime(CLOCK_MONOTONIC,&t0);
    memset(g_tok_count,0,sizeof(g_tok_count));

    /* ── Profiling accumulators ── */
    struct timespec _tp0, _tp1;
    double prof_combined = 0, prof_cdf_ac = 0, prof_fwd = 0;
    double prof_ngram_upd = 0, prof_train = 0;

    for(int i=0;i<n_toks;i++){
        if(clen==0) memcpy(chunk_st,state,sizeof(State));

        float *p_ssm = g_tc_probs;
        clock_gettime(CLOCK_MONOTONIC, &_tp0);
        compute_combined_probs(has_lg, prev1, prev2, prev3, prev4, prev5, prev6, prev7, last_lg, p_ssm, ve, toks, i);
        clock_gettime(CLOCK_MONOTONIC, &_tp1);
        prof_combined += (_tp1.tv_sec-_tp0.tv_sec)+(_tp1.tv_nsec-_tp0.tv_nsec)*1e-9;

        clock_gettime(CLOCK_MONOTONIC, &_tp0);
        int32_t total=probs_to_cdf(p_ssm);
        int tv=toks[i];

        double pm=(double)p_ssm[tv]; if(pm<1e-30)pm=1e-30;
        total_bits += -log2(pm);

        aenc_encode(&enc,g_cdf,tv,total);
        clock_gettime(CLOCK_MONOTONIC, &_tp1);
        prof_cdf_ac += (_tp1.tv_sec-_tp0.tv_sec)+(_tp1.tv_nsec-_tp0.tv_nsec)*1e-9;

        clock_gettime(CLOCK_MONOTONIC, &_tp0);
        forward_token(params,state,tv,last_lg,cache);
        clock_gettime(CLOCK_MONOTONIC, &_tp1);
        prof_fwd += (_tp1.tv_sec-_tp0.tv_sec)+(_tp1.tv_nsec-_tp0.tv_nsec)*1e-9;

        has_lg=1;
        g_tok_count[tv]++;
        g_tokens_seen++;

        clock_gettime(CLOCK_MONOTONIC, &_tp0);
        if (prev1 >= 0) bigram_update(prev1, tv, ve);
        trigram_update(prev2, prev1, tv, ve);
        fourgram_update(prev3, prev2, prev1, tv, ve);
        fivegram_update(prev4, prev3, prev2, prev1, tv, ve);
        sixgram_update(prev5, prev4, prev3, prev2, prev1, tv);
        sevengram_update(prev6, prev5, prev4, prev3, prev2, prev1, tv);
        eightgram_update(prev7, prev6, prev5, prev4, prev3, prev2, prev1, tv);
        sixteengram_update(toks, i, tv);
        thirty2gram_update(toks, i, tv);
        if (prev1 >= 0 && prev2 >= 0) lz_update(prev2, prev1, tv);
        recency_add(tv);
        clock_gettime(CLOCK_MONOTONIC, &_tp1);
        prof_ngram_upd += (_tp1.tv_sec-_tp0.tv_sec)+(_tp1.tv_nsec-_tp0.tv_nsec)*1e-9;

        prev7 = prev6;
        prev6 = prev5;
        prev5 = prev4;
        prev4 = prev3;
        prev3 = prev2;
        prev2 = prev1;
        prev1 = tv;

        /* N-gram count decay */
        if (i+1 >= next_decay) {
            bigram_decay(ve);
            trigram_decay();
            fourgram_decay();
            fivegram_decay();
            sixgram_decay();
            sevengram_decay();
            eightgram_decay();
            next_decay += NGRAM_DECAY_INTERVAL;
        }

        cbuf[clen++]=tv;

        if(clen>=CHUNK_SZ || i==n_toks-1){
            int iters = (chunk_num < 10) ? 8 :
                        (chunk_num < 30) ? 4 : TRAIN_ITERS;
            clock_gettime(CLOCK_MONOTONIC, &_tp0);
            if(clen>=2)
                train_chunk(params,am,av,grads,cbuf,clen,chunk_st,&astep,state,last_lg,iters);
            clock_gettime(CLOCK_MONOTONIC, &_tp1);
            prof_train += (_tp1.tv_sec-_tp0.tv_sec)+(_tp1.tv_nsec-_tp0.tv_nsec)*1e-9;
            clen=0;
            chunk_num++;
            /* LR decay */
            g_lr = ADAM_LR / (1.0f + chunk_num * LR_DECAY_RATE);
        }

        if((i+1)%2000==0){
            clock_gettime(CLOCK_MONOTONIC,&t1);
            double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
            double bpt_now = total_bits/(i+1);
            double bpb_est = bpt_now * ((double)n_toks / N_orig);
            fprintf(stderr,"\r  [%d/%d] %.3f bpt  %.3f bpb(est)  %.0f tok/s",
                    i+1, n_toks, bpt_now, bpb_est, (i+1)/el);
        }
    }

    /* ── Print profiling results ── */
    {
        double prof_total = prof_combined + prof_cdf_ac + prof_fwd + prof_ngram_upd + prof_train;
        fprintf(stderr, "\n  PROFILE (%.1fs total):\n", prof_total);
        fprintf(stderr, "    train_chunk:   %6.1fs  %5.1f%%\n", prof_train,   100*prof_train/prof_total);
        fprintf(stderr, "    forward_tok:   %6.1fs  %5.1f%%\n", prof_fwd,     100*prof_fwd/prof_total);
        fprintf(stderr, "    combined_prob: %6.1fs  %5.1f%%\n", prof_combined,100*prof_combined/prof_total);
        fprintf(stderr, "    cdf+arith:     %6.1fs  %5.1f%%\n", prof_cdf_ac,  100*prof_cdf_ac/prof_total);
        fprintf(stderr, "    ngram_update:  %6.1fs  %5.1f%%\n", prof_ngram_upd,100*prof_ngram_upd/prof_total);
    }
    aenc_finish(&enc);

    clock_gettime(CLOCK_MONOTONIC,&t1);
    double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    fprintf(stderr,"\r                                                                   \r");

    double mean_d = (double)(g_imap[g_vocab_eff-1]) / (g_vocab_eff > 1 ? g_vocab_eff-1 : 1);
    int rice_k = 0;
    while ((1<<(rice_k+1)) < (int)(mean_d + 0.5) && rice_k < 15) rice_k++;
    uint8_t *ibuf = malloc((size_t)g_vocab_eff * 20 + 64);
    ibuf[0] = (uint8_t)rice_k;
    BitW bw; bw_init(&bw, ibuf+1);
    { uint32_t prev = 0;
      for (int i = 0; i < g_vocab_eff; i++) {
          bw_rice(&bw, g_imap[i] - prev, rice_k);
          prev = g_imap[i];
      }
    }
    bw_flush(&bw);
    size_t ilen = 1 + (size_t)bw.nbytes;
    uint32_t ve32 = (uint32_t)g_vocab_eff;
    uint32_t imap_bytes = (uint32_t)ilen;
    FILE *fo=fopen(outpath,"wb");
    fwrite("SSM6",1,4,fo);
    fwrite(&N_orig,8,1,fo);
    fwrite(&N_toks,8,1,fo);
    fwrite(&ve32,4,1,fo);
    fwrite(&imap_bytes,4,1,fo);
    fwrite(ibuf,1,ilen,fo);
    fwrite(enc.buf,1,enc.len,fo);
    fclose(fo);
    free(ibuf);

    uint64_t csz=4+8+8+4+4+(uint64_t)ilen+enc.len+pp_overhead;
    printf("Output: %s (%lu bytes, imap=%zuB Rice(k=%d) vs %dB flat, pp=%zuB)\n",outpath,
           (unsigned long)csz, ilen, rice_k, g_vocab_eff*4, pp_overhead);
    printf("Ratio: %.4f  (%.3f bpb from %.3f bpt × %.4f tok/byte)\n",
           (double)csz/N_orig, total_bits/N_orig, total_bits/n_toks, (double)n_toks/N_orig);
    printf("Time: %.1fs compress (%.0f tok/s)\n", el, n_toks/el);

    free(toks);free(last_lg);free(params);free(am);free(av);free(grads);
    free(state);free(chunk_st);free(cache);free(enc.buf);
    bigram_free(ve); trigram_free(); fourgram_free(); fivegram_free();
    sixgram_free(); sevengram_free(); eightgram_free();
    sixteengram_free(); thirty2gram_free();
    lz_free();
    return 0;
}

/* ── Decompress ── */
int do_decompress(const char *inpath, const char *outpath) {
    FILE *f=fopen(inpath,"rb");
    if(!f){perror(inpath);return 1;}
    fseek(f,0,SEEK_END); long fsize=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t *comp=malloc(fsize);
    fread(comp,1,fsize,f); fclose(f);

    if(memcmp(comp,"SSM6",4)!=0){fprintf(stderr,"Not SSM6 file\n");free(comp);return 1;}
    uint64_t N_orig, N_toks;
    memcpy(&N_orig,comp+4,8);
    memcpy(&N_toks,comp+12,8);
    uint32_t ve32; memcpy(&ve32,comp+20,4);
    g_vocab_eff = (int)ve32;
    uint32_t imap_bytes; memcpy(&imap_bytes,comp+24,4);
    memset(g_remap,-1,sizeof(g_remap));
    { int k = (int)comp[28];
      BitR br; br_init(&br, comp+29);
      uint32_t prev = 0;
      for (int i = 0; i < g_vocab_eff; i++) {
          prev += br_rice(&br, k);
          g_imap[i] = prev;
          g_remap[prev] = (int32_t)i;
      }
    }
    size_t hdr_sz = 4+8+8+4+4+(size_t)imap_bytes;
    printf("Input: %s (%ld bytes), orig: %lu bytes, %lu tokens, vocab=%d\n",
           inpath,fsize,(unsigned long)N_orig,(unsigned long)N_toks,g_vocab_eff);

    size_t params_sz = ALIGN32_SZ(sizeof(Params));
    size_t state_sz  = ALIGN32_SZ(sizeof(State));
    size_t scache_sz = ALIGN32_SZ(sizeof(SCache));
    Params *params = aligned_alloc(32, params_sz); memset(params, 0, params_sz);
    Params *am     = aligned_alloc(32, params_sz); memset(am,     0, params_sz);
    Params *av     = aligned_alloc(32, params_sz); memset(av,     0, params_sz);
    Params *grads  = aligned_alloc(32, params_sz); memset(grads,  0, params_sz);
    init_params(params);
    recompute_A_neg(params);

    State  *state    = aligned_alloc(32, state_sz);  memset(state,    0, state_sz);
    State  *chunk_st = aligned_alloc(32, state_sz);  memset(chunk_st, 0, state_sz);
    SCache *cache    = aligned_alloc(32, scache_sz); memset(cache,    0, scache_sz);
    ADec dec; adec_init(&dec,comp+hdr_sz,fsize-hdr_sz);

    int32_t *toks=malloc(N_toks*sizeof(int32_t));
    int32_t cbuf[CHUNK_SZ];
    int clen=0, astep=0, has_lg=0, chunk_num=0;
    float *last_lg=calloc(g_vocab_eff,sizeof(float));

    int ve = g_vocab_eff;
    bigram_init(ve);
    lz_init();
    trigram_init();
    fourgram_init();
    fivegram_init();
    sixgram_init();
    sevengram_init();
    eightgram_init();
    sixteengram_init();
    thirty2gram_init();
    recency_init();
    g_lr = ADAM_LR;
    g_tokens_seen = 0;
    int prev1=-1, prev2=-1, prev3=-1, prev4=-1, prev5=-1, prev6=-1, prev7=-1;
    int next_decay = NGRAM_DECAY_INTERVAL > 0 ? NGRAM_DECAY_INTERVAL : INT32_MAX;

    struct timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    memset(g_tok_count,0,sizeof(g_tok_count));

    for(uint64_t i=0;i<N_toks;i++){
        if(clen==0) memcpy(chunk_st,state,sizeof(State));

        float *p_ssm = g_tc_probs;
        compute_combined_probs(has_lg, prev1, prev2, prev3, prev4, prev5, prev6, prev7, last_lg, p_ssm, ve, toks, i);

        int32_t total=probs_to_cdf(p_ssm);
        int tv=adec_decode(&dec,g_cdf,total);
        toks[i]=(int32_t)tv;

        forward_token(params,state,tv,last_lg,cache);
        has_lg=1;
        g_tok_count[tv]++;
        g_tokens_seen++;
        if (prev1 >= 0) bigram_update(prev1, tv, ve);
        trigram_update(prev2, prev1, tv, ve);
        fourgram_update(prev3, prev2, prev1, tv, ve);
        fivegram_update(prev4, prev3, prev2, prev1, tv, ve);
        sixgram_update(prev5, prev4, prev3, prev2, prev1, tv);
        sevengram_update(prev6, prev5, prev4, prev3, prev2, prev1, tv);
        eightgram_update(prev7, prev6, prev5, prev4, prev3, prev2, prev1, tv);
        sixteengram_update(toks, (int)i, tv);
        thirty2gram_update(toks, (int)i, tv);
        if (prev1 >= 0 && prev2 >= 0) lz_update(prev2, prev1, tv);
        recency_add(tv);
        prev7 = prev6;
        prev6 = prev5;
        prev5 = prev4;
        prev4 = prev3;
        prev3 = prev2;
        prev2 = prev1;
        prev1 = tv;

        /* N-gram count decay */
        if ((int)(i+1) >= next_decay) {
            bigram_decay(ve);
            trigram_decay();
            fourgram_decay();
            fivegram_decay();
            sixgram_decay();
            sevengram_decay();
            eightgram_decay();
            next_decay += NGRAM_DECAY_INTERVAL;
        }

        cbuf[clen++]=(int32_t)tv;

        if(clen>=CHUNK_SZ || i==N_toks-1){
            int iters = (chunk_num < 10) ? 8 :
                        (chunk_num < 30) ? 4 : TRAIN_ITERS;
            if(clen>=2)
                train_chunk(params,am,av,grads,cbuf,clen,chunk_st,&astep,state,last_lg,iters);
            clen=0;
            chunk_num++;
            /* LR decay */
            g_lr = ADAM_LR / (1.0f + chunk_num * LR_DECAY_RATE);
        }

        if((i+1)%2000==0){
            clock_gettime(CLOCK_MONOTONIC,&t1);
            double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
            fprintf(stderr,"\r  [%lu/%lu] %.1fs  %.0f tok/s",
                    (unsigned long)(i+1),(unsigned long)N_toks,el,(i+1)/el);
        }
    }

    clock_gettime(CLOCK_MONOTONIC,&t1);
    double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    fprintf(stderr,"\r                                                          \r");

    uint8_t *out=malloc(N_orig+1024);
    int n_bytes=detokenize(toks,N_toks,out);
    printf("Detokenized: %d bytes\n",n_bytes);

    FILE *fo=fopen(outpath,"wb");
    fwrite(out,1,n_bytes,fo); fclose(fo);
    printf("Output: %s\n",outpath);
    printf("Time: %.1fs (%.0f tok/s)\n",el,N_toks/el);

    free(comp);free(toks);free(out);free(last_lg);
    free(params);free(am);free(av);free(grads);
    free(state);free(chunk_st);free(cache);
    bigram_free(ve); trigram_free(); fourgram_free(); fivegram_free();
    sixgram_free(); sevengram_free(); eightgram_free();
    sixteengram_free(); thirty2gram_free();
    lz_free();
    return 0;
}

/* ── Verify ── */
int do_verify(const char *inpath) {
    const char *tmp_c="/tmp/ssm_t11_verify.ssm";
    const char *tmp_d="/tmp/ssm_t11_verify.dec";

    FILE *f=fopen(inpath,"rb");
    if(!f){perror(inpath);return 1;}
    fseek(f,0,SEEK_END); long fsize=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t *orig=malloc(fsize); fread(orig,1,fsize,f); fclose(f);

    printf("=== Compress ===\n");
    if(do_compress(inpath,tmp_c)) return 1;
    printf("\n=== Decompress ===\n");
    if(do_decompress(tmp_c,tmp_d)) return 1;

    FILE *fd=fopen(tmp_d,"rb");
    fseek(fd,0,SEEK_END); long dsize=ftell(fd); fseek(fd,0,SEEK_SET);
    uint8_t *dec=malloc(dsize); fread(dec,1,dsize,fd); fclose(fd);

    int ok=1;
    if(fsize!=dsize){ printf("\nMISMATCH: sizes %ld vs %ld\n",fsize,dsize); ok=0; }
    else {
        for(long i=0;i<fsize;i++){
            if(orig[i]!=dec[i]){
                printf("\nMISMATCH at byte %ld: 0x%02x vs 0x%02x\n",i,orig[i],dec[i]);
                ok=0; break;
            }
        }
    }
    if(ok) printf("\nVERIFIED: perfect match!\n");
    free(orig); free(dec);
    return ok?0:1;
}
