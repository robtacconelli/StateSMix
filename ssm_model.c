#include "ssm_model.h"
#include "ssm_math.h"
#include "ssm_vocab.h"
#include <immintrin.h>
#include <omp.h>

/* ── Global buffers ── */
__attribute__((aligned(32))) float g_A_neg[NL][DI * DS];
__attribute__((aligned(32))) float g_tc_logits[VOCAB];
__attribute__((aligned(32))) float g_tc_probs[VOCAB];
__attribute__((aligned(32))) float g_tc_dlogits[VOCAB];
__attribute__((aligned(32))) int32_t g_cdf[VOCAB+1];
int    g_tok_count[VOCAB];
__attribute__((aligned(32))) float  g_mixed_lg[VOCAB];
float g_lr = ADAM_LR;

/* ── AVX2 horizontal sum of 8-lane float vector ── */
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

/* ── AVX2 dot product: DM floats (DM/8 AVX2 registers) ── */
static inline float dot_dm(const float * __restrict__ a, const float * __restrict__ b) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < DM; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
    return hsum256(acc);
}

void recompute_A_neg(const Params *p) {
    for (int l = 0; l < NL; l++)
        for (int idx = 0; idx < DI*DS; idx++)
            g_A_neg[l][idx] = -fast_expf_a(fminf(p->L[l].A_log[idx], 30.0f));
}

void init_params(Params *p) {
    memset(p, 0, sizeof(*p));
    g_rng = 123456789012345ULL;
    float sc = 0.02f;
    for (int i=0; i<g_vocab_eff*DM; i++) p->emb[i]=randn()*sc;
    for (int l=0; l<NL; l++) {
        LParams *lp = &p->L[l];
        for (int i=0;i<DM;i++){lp->ln_g[i]=1.0f; lp->ln_b[i]=0.0f;}
        for (int i=0;i<DM*2*DI;i++) lp->in_w[i]=randn()*sc;
        for (int i=0;i<DI*DC;i++) lp->conv_w[i]=randn()*sc;
        for (int i=0;i<DI*XP_DIM;i++) lp->xp_w[i]=randn()*sc;
        for (int i=0;i<DI;i++) lp->dt_w[i]=randn()*sc;
        for (int i=0;i<DI;i++) lp->dt_b[i]=randn()*0.1f;
        for (int i=0;i<DI;i++)
            for (int j=0;j<DS;j++)
                lp->A_log[i*DS+j]=(float)log((double)(j+1));
        for (int i=0;i<DI;i++) lp->D[i]=1.0f;
        for (int i=0;i<DI*DM;i++) lp->out_w[i]=randn()*sc;
    }
    for (int i=0;i<DM;i++){p->fn_g[i]=1.0f; p->fn_b[i]=0.0f;}
    for (int i=0; i<g_vocab_eff*DM; i++) p->head_w[i]=randn()*sc;
}

void forward_token(const Params * __restrict__ p, State * __restrict__ st, int bv,
                   float * __restrict__ logits, SCache * __restrict__ c) {
    int i,j,k;
    float x[DM];
    c->bv = bv;
    memcpy(x, &p->emb[bv*DM], DM*sizeof(float));

    for (int l=0; l<NL; l++) {
        const LParams *lp = &p->L[l];
        LCache *lc = &c->lc[l];
        float res[DM];
        memcpy(lc->x_pre, x, DM*sizeof(float));
        memcpy(res, x, DM*sizeof(float));

        float mu=0; for(i=0;i<DM;i++) mu+=x[i]; mu/=DM;
        float var=0; for(i=0;i<DM;i++) var+=(x[i]-mu)*(x[i]-mu); var/=DM;
        lc->inv_std = 1.0f/sqrtf(var+1e-5f);
        for(i=0;i<DM;i++){
            lc->x_hat[i]=(x[i]-mu)*lc->inv_std;
            lc->x_norm[i]=lp->ln_g[i]*lc->x_hat[i]+lp->ln_b[i];
        }

        /* in_w projection: [DM] -> [2*DI=128]
         * Loop: for k in DM: y[j] += x_norm[k] * in_w[k*(2*DI) + j]
         * Rewritten as: accumulate outer product in AVX2 chunks of 8
         * 2*DI = 128 = 16 AVX2 registers per row; DM=32 rows */
        { float y[2*DI];
          memset(y, 0, sizeof(y));
          for(k=0; k<DM; k++){
              float xnk = lc->x_norm[k];
              const float *row = lp->in_w + k*(2*DI);
              __m256 vxnk = _mm256_set1_ps(xnk);
              /* 2*DI = 128 floats = 16 AVX2 chunks */
              for(j=0; j<2*DI; j+=8){
                  __m256 vy = _mm256_loadu_ps(y+j);
                  __m256 vr = _mm256_loadu_ps(row+j);
                  vy = _mm256_fmadd_ps(vxnk, vr, vy);
                  _mm256_storeu_ps(y+j, vy);
              }
          }
          memcpy(lc->xs, y, DI*sizeof(float));
          memcpy(lc->z, y+DI, DI*sizeof(float));
        }

        for(i=0;i<DI;i++){
            for(j=0;j<DC-1;j++) lc->full[i*DC+j]=st->cb[l][i*(DC-1)+j];
            lc->full[i*DC+DC-1]=lc->xs[i];
        }
        for(i=0;i<DI;i++){
            float v=lp->conv_b[i];
            for(j=0;j<DC;j++) v+=lc->full[i*DC+j]*lp->conv_w[i*DC+j];
            lc->xc_pre[i]=v; lc->sig_xc[i]=fast_sigmoid(v); lc->xc[i]=v*lc->sig_xc[i];
        }
        for(i=0;i<DI;i++){
            for(j=0;j<DC-2;j++) st->cb[l][i*(DC-1)+j]=st->cb[l][i*(DC-1)+j+1];
            st->cb[l][i*(DC-1)+DC-2]=lc->xs[i];
        }

        /* xp_w projection: [DI=64] -> [XP_DIM=33]
         * XP_DIM=33 is not a multiple of 8, handle with scalar tail */
        { float proj[XP_DIM]; memset(proj,0,sizeof(proj));
          /* XP_DIM = 2*DS+1 = 33; use scalar since not multiple of 8 */
          for(k=0;k<DI;k++){
              float xck=lc->xc[k]; const float *row=lp->xp_w+k*XP_DIM;
              for(j=0;j<XP_DIM;j++) proj[j]+=xck*row[j];
          }
          memcpy(lc->B,proj,DS*sizeof(float));
          memcpy(lc->C,proj+DS,DS*sizeof(float));
          lc->dt_in=proj[2*DS]; }

        for(i=0;i<DI;i++){
            float v=lc->dt_in*lp->dt_w[i]+lp->dt_b[i];
            lc->dt_pre[i]=v; lc->dt[i]=fast_softplus(v);
        }

        memcpy(lc->h_old,st->h[l],DI*DS*sizeof(float));
        for(i=0;i<DI;i++){
            float yi=0, dti=lc->dt[i];
            const float *A_row=g_A_neg[l]+i*DS;
            float *dA_row=lc->dA_cache+i*DS;
            for(j=0;j<DS;j++){
                float A=A_row[j], dA=fast_expf_a(dti*A);
                dA_row[j]=dA;
                int idx=i*DS+j;
                float hn=dA*lc->h_old[idx]+dti*lc->B[j]*lc->xc[i];
                lc->h_new[idx]=hn; st->h[l][idx]=hn; yi+=hn*lc->C[j];
            }
            lc->y[i]=yi+lp->D[i]*lc->xc[i];
        }

        for(i=0;i<DI;i++){
            lc->sig_z[i]=fast_sigmoid(lc->z[i]);
            lc->gate[i]=lc->z[i]*lc->sig_z[i];
            lc->out_pre[i]=lc->y[i]*lc->gate[i];
        }

        /* out_w projection: accumulate [DI] -> [DM]
         * for k in DI: x[i] += out_pre[k] * out_w[k*DM + i] */
        memcpy(x, res, DM*sizeof(float));
        for(k=0; k<DI; k++){
            float opk = lc->out_pre[k];
            const float *row = lp->out_w + k*DM;
            __m256 vopk = _mm256_set1_ps(opk);
            for(j=0; j<DM; j+=8){
                __m256 vx = _mm256_loadu_ps(x+j);
                vx = _mm256_fmadd_ps(vopk, _mm256_loadu_ps(row+j), vx);
                _mm256_storeu_ps(x+j, vx);
            }
        }
    }

    memcpy(c->x_pfn,x,DM*sizeof(float));
    float mu=0; for(i=0;i<DM;i++) mu+=x[i]; mu/=DM;
    float var=0; for(i=0;i<DM;i++) var+=(x[i]-mu)*(x[i]-mu); var/=DM;
    c->inv_std_fn=1.0f/sqrtf(var+1e-5f);
    for(i=0;i<DM;i++){
        c->x_hat_fn[i]=(x[i]-mu)*c->inv_std_fn;
        c->x_final[i]=p->fn_g[i]*c->x_hat_fn[i]+p->fn_b[i];
    }

    /* Head logits: [g_vocab_eff] dot products of DM=32 floats each */
    {
        const float *xf = c->x_final;
        #pragma omp parallel for schedule(static)
        for(j=0; j<g_vocab_eff; j++){
            logits[j] = dot_dm(xf, p->head_w + j*DM);
        }
    }
}

void backward_token(const Params * __restrict__ p, const SCache * __restrict__ c,
                    const float * __restrict__ dlogits, Params * __restrict__ g) {
    int i,j,k;
    const float inv_N=1.0f/DM;

    /* Head backward: accumulate dx_f[DM] and update grow[DM]
     * Parallelized: each thread accumulates local dx_f, then reduce */
    float dx_f[DM];
    {
        __m256 vdx[DM/8];
        for(i=0; i<DM/8; i++) vdx[i] = _mm256_setzero_ps();

        #pragma omp parallel
        {
            __m256 local_vdx[DM/8];
            for(int ii=0; ii<DM/8; ii++) local_vdx[ii] = _mm256_setzero_ps();

            #pragma omp for schedule(static) nowait
            for(j=0; j<g_vocab_eff; j++){
                float dl = dlogits[j];
                __m256 vdl = _mm256_set1_ps(dl);
                const float *row  = p->head_w + j*DM;
                float       *grow = g->head_w + j*DM;
                for(int ii=0; ii<DM; ii+=8){
                    local_vdx[ii/8] = _mm256_fmadd_ps(_mm256_loadu_ps(row+ii), vdl, local_vdx[ii/8]);
                    __m256 vg = _mm256_loadu_ps(grow+ii);
                    vg = _mm256_fmadd_ps(_mm256_loadu_ps(c->x_final+ii), vdl, vg);
                    _mm256_storeu_ps(grow+ii, vg);
                }
            }
            #pragma omp critical
            {
                for(int ii=0; ii<DM/8; ii++)
                    vdx[ii] = _mm256_add_ps(vdx[ii], local_vdx[ii]);
            }
        }
        for(i=0; i<DM; i+=8)
            _mm256_storeu_ps(dx_f+i, vdx[i/8]);
    }

    for(i=0;i<DM;i++){g->fn_g[i]+=dx_f[i]*c->x_hat_fn[i]; g->fn_b[i]+=dx_f[i];}
    float dxh_fn[DM];
    for(i=0;i<DM;i++) dxh_fn[i]=dx_f[i]*p->fn_g[i];
    float s1=0,s2=0;
    for(i=0;i<DM;i++){s1+=dxh_fn[i]; s2+=dxh_fn[i]*c->x_hat_fn[i];}
    float dx[DM];
    for(i=0;i<DM;i++)
        dx[i]=c->inv_std_fn*(dxh_fn[i]-s1*inv_N-c->x_hat_fn[i]*s2*inv_N);

    for(int l=NL-1;l>=0;l--){
        const LParams *lp=&p->L[l];
        LParams *lg=&g->L[l];
        const LCache *lc=&c->lc[l];

        /* out_w backward:
         * d_op[i] = sum_j(out_w[i*DM+j] * dx[j])  -- dot product DM
         * lg->out_w[i*DM+j] += out_pre[i] * dx[j]  -- outer product update */
        float d_op[DI];
        {
            for(i=0; i<DI; i++){
                const float *row = lp->out_w + i*DM;
                float       *gw  = lg->out_w + i*DM;
                float opi = lc->out_pre[i];
                __m256 vopi = _mm256_set1_ps(opi);

                /* dot product row·dx + gradient update */
                __m256 acc = _mm256_setzero_ps();
                for(j=0; j<DM; j+=8){
                    __m256 vdx_j = _mm256_loadu_ps(dx+j);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(row+j), vdx_j, acc);
                    __m256 vg = _mm256_loadu_ps(gw+j);
                    vg = _mm256_fmadd_ps(vopi, vdx_j, vg);
                    _mm256_storeu_ps(gw+j, vg);
                }
                d_op[i] = hsum256(acc);
            }
        }

        float d_y[DI],d_z[DI];
        for(i=0;i<DI;i++){
            d_y[i]=d_op[i]*lc->gate[i];
            float dg=d_op[i]*lc->y[i];
            d_z[i]=dg*lc->sig_z[i]*(1.0f+lc->z[i]*(1.0f-lc->sig_z[i]));
        }

        float d_xc[DI],d_C[DS];
        for(i=0;i<DI;i++){lg->D[i]+=d_y[i]*lc->xc[i]; d_xc[i]=d_y[i]*lp->D[i];}
        memset(d_C,0,DS*sizeof(float));
        for(i=0;i<DI;i++){
            float dyi=d_y[i]; const float *row=lc->h_new+i*DS;
            for(j=0;j<DS;j++) d_C[j]+=row[j]*dyi;
        }
        float d_hn[DI*DS];
        for(i=0;i<DI;i++) for(j=0;j<DS;j++) d_hn[i*DS+j]=d_y[i]*lc->C[j];

        float d_dt[DI],d_B[DS];
        memset(d_B,0,DS*sizeof(float));
        for(i=0;i<DI;i++){
            float dt_sum=0;
            const float *A_row=g_A_neg[l]+i*DS;
            const float *dA_row=lc->dA_cache+i*DS;
            for(j=0;j<DS;j++){
                float A=A_row[j], dA_v=dA_row[j];
                int idx=i*DS+j;
                float d_dB=d_hn[idx]*lc->xc[i];
                dt_sum+=d_dB*lc->B[j];
                d_B[j]+=d_dB*lc->dt[i];
                d_xc[i]+=d_hn[idx]*lc->dt[i]*lc->B[j];
                float d_dtA=d_hn[idx]*lc->h_old[idx]*dA_v;
                dt_sum+=d_dtA*A;
                lg->A_log[idx]+=d_dtA*lc->dt[i]*A;
            }
            d_dt[i]=dt_sum;
        }

        float d_dt_in=0;
        for(i=0;i<DI;i++){
            float sig=fast_sigmoid(lc->dt_pre[i]);
            float d_dtp=d_dt[i]*sig;
            lg->dt_w[i]+=lc->dt_in*d_dtp;
            lg->dt_b[i]+=d_dtp;
            d_dt_in+=d_dtp*lp->dt_w[i];
        }

        /* xp_w backward: scalar (XP_DIM=33 not multiple of 8) */
        float d_proj[XP_DIM];
        memcpy(d_proj,d_B,DS*sizeof(float));
        memcpy(d_proj+DS,d_C,DS*sizeof(float));
        d_proj[2*DS]=d_dt_in;
        for(i=0;i<DI;i++){
            float xci=lc->xc[i], v=0;
            const float *row=lp->xp_w+i*XP_DIM;
            float *grow=lg->xp_w+i*XP_DIM;
            for(j=0;j<XP_DIM;j++){v+=row[j]*d_proj[j]; grow[j]+=xci*d_proj[j];}
            d_xc[i]+=v;
        }

        float d_xcp[DI];
        for(i=0;i<DI;i++)
            d_xcp[i]=d_xc[i]*lc->sig_xc[i]*(1.0f+lc->xc_pre[i]*(1.0f-lc->sig_xc[i]));

        for(i=0;i<DI;i++) lg->conv_b[i]+=d_xcp[i];
        float d_xs[DI];
        for(i=0;i<DI;i++){
            for(j=0;j<DC;j++) lg->conv_w[i*DC+j]+=d_xcp[i]*lc->full[i*DC+j];
            d_xs[i]=d_xcp[i]*lp->conv_w[i*DC+DC-1];
        }

        /* in_w backward:
         * d_xz[2*DI] = [d_xs || d_z]
         * d_xn[k] = sum_j(in_w[k*(2*DI)+j] * d_xz[j])  -- dot product 2*DI=128
         * lg->in_w[k*(2*DI)+j] += x_norm[k] * d_xz[j]  -- gradient update
         * 2*DI=128 = 16 AVX2 registers; DM=32 rows */
        float d_xz[2*DI];
        memcpy(d_xz, d_xs, DI*sizeof(float));
        memcpy(d_xz+DI, d_z, DI*sizeof(float));
        float d_xn[DM];
        for(k=0; k<DM; k++){
            float xnk = lc->x_norm[k];
            const float *row  = lp->in_w + k*(2*DI);
            float       *grow = lg->in_w + k*(2*DI);
            __m256 vxnk = _mm256_set1_ps(xnk);
            __m256 vv = _mm256_setzero_ps();
            for(j=0; j<2*DI; j+=8){
                __m256 vr   = _mm256_loadu_ps(row+j);
                __m256 vdxz = _mm256_loadu_ps(d_xz+j);
                vv = _mm256_fmadd_ps(vr, vdxz, vv);
                __m256 vg = _mm256_loadu_ps(grow+j);
                vg = _mm256_fmadd_ps(vxnk, vdxz, vg);
                _mm256_storeu_ps(grow+j, vg);
            }
            d_xn[k] = hsum256(vv);
        }

        for(i=0;i<DM;i++){lg->ln_g[i]+=d_xn[i]*lc->x_hat[i]; lg->ln_b[i]+=d_xn[i];}
        float dxh[DM];
        for(i=0;i<DM;i++) dxh[i]=d_xn[i]*lp->ln_g[i];
        float ss1=0,ss2=0;
        for(i=0;i<DM;i++){ss1+=dxh[i]; ss2+=dxh[i]*lc->x_hat[i];}
        float dx_ln[DM];
        for(i=0;i<DM;i++)
            dx_ln[i]=lc->inv_std*(dxh[i]-ss1*inv_N-lc->x_hat[i]*ss2*inv_N);
        for(i=0;i<DM;i++) dx[i]+=dx_ln[i];
    }

    int bv=c->bv;
    for(i=0;i<DM;i++) g->emb[bv*DM+i]+=dx[i];
}

void do_softmax(const float * __restrict__ logits, float * __restrict__ probs) {
    int ve = g_vocab_eff;
    float mx = logits[0];
    #pragma omp parallel for reduction(max:mx) schedule(static)
    for(int i=1;i<ve;i++) if(logits[i]>mx) mx=logits[i];
    float sum=0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for(int i=0;i<ve;i++){probs[i]=fast_expf_v(logits[i]-mx); sum+=probs[i];}
    float inv=1.0f/sum;
    #pragma omp parallel for schedule(static)
    for(int i=0;i<ve;i++) probs[i]*=inv;
}

int32_t probs_to_cdf(const float * __restrict__ probs) {
    int ve = g_vocab_eff;
    g_cdf[0]=0;
    for(int i=0;i<ve;i++){
        int32_t f=(int32_t)(probs[i]*AC_SCALE+0.5f);
        if(f<1)f=1;
        g_cdf[i+1]=g_cdf[i]+f;
    }
    return g_cdf[ve];
}

void adam_update(Params * __restrict__ params, Params * __restrict__ grads,
                 Params * __restrict__ am, Params * __restrict__ av, int n, int step) {
    float *p=(float*)params,*gr=(float*)grads,*m=(float*)am,*v=(float*)av;
    float inv_n=1.0f/n, gnorm=0;
    int ve = g_vocab_eff;
    int emb_off  = FIXED_FLOATS;
    int head_off = FIXED_FLOATS + VOCAB*DM;
    for(int i=0;i<FIXED_FLOATS;i++){float gi=gr[i]*inv_n;gnorm+=gi*gi;}
    float gnorm_emb=0, gnorm_head=0;
    #pragma omp parallel for reduction(+:gnorm_emb) schedule(static)
    for(int i=emb_off;i<emb_off+ve*DM;i++){float gi=gr[i]*inv_n;gnorm_emb+=gi*gi;}
    #pragma omp parallel for reduction(+:gnorm_head) schedule(static)
    for(int i=head_off;i<head_off+ve*DM;i++){float gi=gr[i]*inv_n;gnorm_head+=gi*gi;}
    gnorm=sqrtf(gnorm+gnorm_emb+gnorm_head);
    float clip=(gnorm>GRAD_CLIP)?GRAD_CLIP/gnorm:1.0f;
    float bc1=1.0f/(1.0f-powf(ADAM_B1,(float)step));
    float bc2=1.0f/(1.0f-powf(ADAM_B2,(float)step));
    float lr=g_lr;
#define ADAM_STEP(idx) do{ \
    float gi=gr[idx]*inv_n*clip; \
    m[idx]=ADAM_B1*m[idx]+(1.0f-ADAM_B1)*gi; \
    v[idx]=ADAM_B2*v[idx]+(1.0f-ADAM_B2)*gi*gi; \
    p[idx]-=lr*(m[idx]*bc1)/(sqrtf(v[idx]*bc2)+ADAM_EPS); }while(0)
    for(int i=0;i<FIXED_FLOATS;i++) ADAM_STEP(i);
    #pragma omp parallel for schedule(static)
    for(int i=emb_off;i<emb_off+ve*DM;i++) ADAM_STEP(i);
    #pragma omp parallel for schedule(static)
    for(int i=head_off;i<head_off+ve*DM;i++) ADAM_STEP(i);
#undef ADAM_STEP
}

void train_chunk(Params * __restrict__ params, Params * __restrict__ am,
                 Params * __restrict__ av, Params * __restrict__ grads,
                 const int32_t * __restrict__ chunk, int clen,
                 const State * __restrict__ st0, int * __restrict__ astep,
                 State * __restrict__ st_out, float * __restrict__ last_lg_out,
                 int train_iters) {
    if(clen<2) return;
    State st; SCache cache;
    int ve = g_vocab_eff;
    for(int iter=0;iter<train_iters;iter++){
        memset(grads,0,sizeof(Params));
        memcpy(&st,st0,sizeof(State));
        for(int i=0;i<clen-1;i++){
            forward_token(params,&st,chunk[i],g_tc_logits,&cache);
            do_softmax(g_tc_logits,g_tc_probs);
            { float eps=0.12f, eps_v=eps/ve;
            for(int j=0;j<ve;j++) g_tc_dlogits[j]=g_tc_probs[j]-eps_v;
            g_tc_dlogits[chunk[i+1]]-=(1.0f-eps); }
            backward_token(params,&cache,g_tc_dlogits,grads);
        }
        (*astep)++;
        adam_update(params,grads,am,av,clen-1,*astep);
        recompute_A_neg(params);
    }
    forward_token(params,&st,chunk[clen-1],g_tc_logits,&cache);
    if(st_out)     memcpy(st_out,&st,sizeof(State));
    if(last_lg_out) memcpy(last_lg_out,g_tc_logits,ve*sizeof(float));
}
