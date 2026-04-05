#ifndef SSM_VOCAB_H
#define SSM_VOCAB_H

#include "ssm_config.h"

/* Compact-vocab: only tokens that appear in the file get modelled.
 * g_vocab_eff  <= VOCAB is set after tokenising.
 * g_remap[orig_id] = compact_id  (or -1 if unseen)
 * g_imap[compact_id] = orig_id   (for writing to the file header) */
extern int      g_vocab_eff;
#define REMAP_MAX (VOCAB + 4096)   /* room for extra BPE super-tokens */
extern int32_t  g_remap[REMAP_MAX]; /* orig -> compact */
extern uint32_t g_imap[VOCAB];      /* compact -> orig  */

/* Build compact vocab from the token sequence.
 * Tokens are sorted by first-occurrence order so the mapping is
 * deterministic and identical on compress / decompress. */
void build_vocab_map(int32_t *toks, int n);

/* Sort imap by orig-ID value (ascending) and rebuild g_remap.
 * Call after build_vocab_map, before remap_toks.
 * Compact IDs become 0..ve-1 in sorted-orig-ID order. */
void sort_vocab_map(void);

/* Remap token sequence in-place (orig IDs -> compact IDs) */
void remap_toks(int32_t *toks, int n);

#endif /* SSM_VOCAB_H */
