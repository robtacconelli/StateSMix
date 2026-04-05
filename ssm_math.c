#include "ssm_math.h"

uint64_t g_rng;

float randn(void) {
    uint64_t s = g_rng;
    s ^= s<<13; s ^= s>>7; s ^= s<<17; g_rng = s;
    double u1 = (double)((s>>12)|1) * (1.0/(double)(1ULL<<52));
    s ^= s<<13; s ^= s>>7; s ^= s<<17; g_rng = s;
    double u2 = (double)(s>>12) * (1.0/(double)(1ULL<<52));
    return (float)(sqrt(-2.0*log(u1)) * cos(6.283185307179586*u2));
}
