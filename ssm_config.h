#ifndef SSM_CONFIG_H
#define SSM_CONFIG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* ── Model dimensions ── */
#define DM      32
#define DS      16
#define DC      4
#define DI      (DM * 2)
#define NL      2
#define VOCAB   49152
#define XP_DIM  (2*DS+1)

/* ── Training hyperparameters ── */
#define CHUNK_SZ    32
#define TRAIN_ITERS 2
#define ADAM_LR     0.002f
#define ADAM_B1     0.9f
#define ADAM_B2     0.999f
#define ADAM_EPS    1e-8f
#define GRAD_CLIP   5.0f

/* ── Arithmetic coder ── */
#define AC_SCALE    (1<<16)   /* must be > VOCAB: 65536 > 49152 */

/* ── Count-based logit mixing ── */
#define COUNT_LAMBDA 0.1f

#endif /* SSM_CONFIG_H */
