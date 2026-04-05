#include "ssm_arithmetic.h"
#include "ssm_vocab.h"

/* ── Encoder internals ── */

static void aenc_bit(AEnc *e, int b) {
    e->cur_byte |= (b << e->bit_pos);
    if (--e->bit_pos < 0) {
        if (e->len >= e->cap) { e->cap*=2; e->buf=realloc(e->buf,e->cap); }
        e->buf[e->len++] = e->cur_byte;
        e->cur_byte=0; e->bit_pos=7;
    }
}

static void aenc_out(AEnc *e, int b) {
    aenc_bit(e,b);
    for (int i=0;i<e->pending;i++) aenc_bit(e,1-b);
    e->pending=0;
}

void aenc_init(AEnc *e) {
    e->lo=0; e->hi=0xFFFFFFFFU; e->pending=0;
    e->cap=131072; e->buf=malloc(e->cap); e->len=0;
    e->cur_byte=0; e->bit_pos=7;
}

void aenc_encode(AEnc *e, const int32_t *cdf, int sym, int32_t total) {
    uint64_t r = (uint64_t)e->hi - e->lo + 1;
    e->hi = e->lo + (uint32_t)((r*(uint32_t)cdf[sym+1])/(uint32_t)total) - 1;
    e->lo = e->lo + (uint32_t)((r*(uint32_t)cdf[sym])/(uint32_t)total);
    for (;;) {
        if (e->hi < 0x80000000U) aenc_out(e,0);
        else if (e->lo >= 0x80000000U) { aenc_out(e,1); e->lo-=0x80000000U; e->hi-=0x80000000U; }
        else if (e->lo>=0x40000000U && e->hi<0xC0000000U) { e->pending++; e->lo-=0x40000000U; e->hi-=0x40000000U; }
        else break;
        e->lo<<=1; e->hi=(e->hi<<1)|1;
    }
}

void aenc_finish(AEnc *e) {
    e->pending++;
    aenc_out(e, e->lo < 0x40000000U ? 0 : 1);
    if (e->bit_pos < 7) {
        if (e->len>=e->cap){e->cap*=2;e->buf=realloc(e->buf,e->cap);}
        e->buf[e->len++]=e->cur_byte;
    }
}

/* ── Decoder internals ── */

static int adec_rb(ADec *d) {
    if (d->bp < d->dlen*8) {
        int by=d->bp/8, bi=7-(d->bp%8); d->bp++;
        return (d->data[by]>>bi)&1;
    }
    return 0;
}

void adec_init(ADec *d, const uint8_t *data, size_t len) {
    d->lo=0; d->hi=0xFFFFFFFFU; d->data=data; d->dlen=len; d->bp=0;
    d->val=0; for(int i=0;i<32;i++) d->val=(d->val<<1)|adec_rb(d);
}

int adec_decode(ADec *d, const int32_t *cdf, int32_t total) {
    uint64_t r=(uint64_t)d->hi-d->lo+1;
    uint64_t sv=((uint64_t)(d->val-d->lo)+1)*(uint32_t)total-1; sv/=r;
    int a=0, b=g_vocab_eff-1;
    while(a<=b){int m=(a+b)/2; if((uint32_t)cdf[m+1]<=sv) a=m+1; else b=m-1;}
    int sym=a;
    d->hi=d->lo+(uint32_t)((r*(uint32_t)cdf[sym+1])/(uint32_t)total)-1;
    d->lo=d->lo+(uint32_t)((r*(uint32_t)cdf[sym])/(uint32_t)total);
    for(;;){
        if(d->hi<0x80000000U){}
        else if(d->lo>=0x80000000U){d->lo-=0x80000000U;d->hi-=0x80000000U;d->val-=0x80000000U;}
        else if(d->lo>=0x40000000U&&d->hi<0xC0000000U){d->lo-=0x40000000U;d->hi-=0x40000000U;d->val-=0x40000000U;}
        else break;
        d->lo<<=1;d->hi=(d->hi<<1)|1;d->val=(d->val<<1)|adec_rb(d);
    }
    return sym;
}
