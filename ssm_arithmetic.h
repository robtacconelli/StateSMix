#ifndef SSM_ARITHMETIC_H
#define SSM_ARITHMETIC_H

#include "ssm_config.h"

/* ── Arithmetic Encoder ── */
typedef struct {
    uint32_t lo, hi;
    int pending;
    uint8_t *buf; size_t len, cap;
    int cur_byte, bit_pos;
} AEnc;

void aenc_init(AEnc *e);
void aenc_encode(AEnc *e, const int32_t *cdf, int sym, int32_t total);
void aenc_finish(AEnc *e);

/* ── Arithmetic Decoder ── */
typedef struct {
    uint32_t lo, hi, val;
    const uint8_t *data; size_t dlen, bp;
} ADec;

void adec_init(ADec *d, const uint8_t *data, size_t len);
int  adec_decode(ADec *d, const int32_t *cdf, int32_t total);

#endif /* SSM_ARITHMETIC_H */
