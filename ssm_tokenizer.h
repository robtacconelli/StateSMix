#ifndef SSM_TOKENIZER_H
#define SSM_TOKENIZER_H

#include "ssm_config.h"

/* Load tokenizer binary from path.
 * Format: "BPET" vocab_size(u32) n_merges(u32)
 *         byte2tok[256](i32) vocab_bytes(len:u8 + data) merges(l,r,m:u32x3)
 * Returns 0 on success, 1 on error. */
int load_tokenizer(const char *path);

/* BPE-encode a byte array into token IDs.
 * Returns number of tokens. Caller must free *out. */
int bpe_encode(const uint8_t *bytes, int n, int32_t **out);

/* Detokenize: compact token IDs -> bytes.
 * toks[] holds compact IDs; maps through g_imap to get original IDs.
 * Returns number of output bytes. */
int detokenize(const int32_t *toks, int n_toks, uint8_t *out);

#endif /* SSM_TOKENIZER_H */
