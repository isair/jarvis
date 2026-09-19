/* Minimal PFFFT-compatible implementation (see pffft.h for the layout).
 * Backward = inverse, conjugate trick with 1/N scaling.
 */
#include "pffft.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define PI 3.14159265358979323846

struct PFFFT_Setup_data {
  int N;
  int type;
  int n;
  float* sc; /* two planes of 2n floats */
};

void* pffft_aligned_malloc(size_t nb_bytes) { return malloc(nb_bytes ? nb_bytes : 1); }
void pffft_aligned_free(void* p) { free(p); }
int pffft_simd_size(void) { return 1; }

static int pow2_of(int v, int* nb) {
  int k = 0, x = 1;
  while (x < v) { x <<= 1; ++k; }
  *nb = k;
  return x == v;
}

/* forward: in(2n) -> out(2n); scratch = the second plane */
static void fft_pow2(const float* in, float* out, float* scratch, int n) {
  int nb = 0;
  pow2_of(n, &nb);
  for (int i = 0; i < n; ++i) {
    int r = 0;
    for (int b = 0; b < nb; ++b) r |= (((i >> b) & 1) << (nb - 1 - b));
    scratch[2 * r] = in[2 * i];
    scratch[2 * r + 1] = in[2 * i + 1];
  }
  memcpy(out, scratch, sizeof(float) * 2u * (unsigned)n);
  for (int ln = 2; ln <= n; ln <<= 1) {
    for (int k = 0; k < n; k += ln) {
      for (int j = 0; j < ln / 2; ++j) {
        const double ang = -2.0 * PI * j / ln;
        const double wr = cos(ang), wi = sin(ang);
        const int i0 = 2 * (k + j), i1 = 2 * (k + j + ln / 2);
        const double x0r = out[i0], x0i = out[i0 + 1];
        const double x1r = out[i1], x1i = out[i1 + 1];
        const double tr = x1r * wr - x1i * wi;
        const double ti = x1r * wi + x1i * wr;
        out[i0] = (float)(x0r + tr);
        out[i0 + 1] = (float)(x0i + ti);
        out[i1] = (float)(x0r - tr);
        out[i1 + 1] = (float)(x0i - ti);
      }
    }
  }
}

static void fft_direct(const float* in, float* out, int n) {
  for (int k = 0; k < n; ++k) {
    double sr = 0.0, si = 0.0;
    for (int t = 0; t < n; ++t) {
      const double ang = -2.0 * PI * (double)((long)k * t % n) / (double)n;
      const double wr = cos(ang), wi = sin(ang);
      sr += (double)in[2 * t] * wr - (double)in[2 * t + 1] * wi;
      si += (double)in[2 * t] * wi + (double)in[2 * t + 1] * wr;
    }
    out[2 * k] = (float)sr;
    out[2 * k + 1] = (float)si;
  }
}

static void cfft_fwd(const float* in, float* out, float* scratch, int n) {
  if (n <= 1) { out[0] = in[0]; out[1] = in[1]; return; }
  int nb = 0;
  if (pow2_of(n, &nb)) fft_pow2(in, out, scratch, n);
  else fft_direct(in, out, n);
}

static void cfft_inv(const float* in, float* out, float* scratch, int n) {
  if (n <= 1) { out[0] = in[0]; out[1] = in[1]; return; }
  const size_t bytes = sizeof(float) * 2u * (unsigned)n;
  float* tmp = (float*)malloc(bytes);
  for (int k = 0; k < n; ++k) { tmp[2 * k] = in[2 * k]; tmp[2 * k + 1] = -in[2 * k + 1]; }
  if (pow2_of(n, (&(int){0}))) {} /* not needed */
  int nb = 0;
  if (pow2_of(n, &nb)) fft_pow2(tmp, out, scratch, n);
  else fft_direct(tmp, out, n);
  for (int k = 0; k < n; ++k) {
    out[2 * k] /= (float)n;
    out[2 * k + 1] = -out[2 * k + 1] / (float)n;
  }
  free(tmp);
}

PFFFT_Setup_data* pffft_new_setup(int N, PFFFT_1D transfo) {
  if (N <= 0) return NULL;
  PFFFT_Setup_data* s = (PFFFT_Setup_data*)calloc(1, sizeof(*s));
  s->N = N;
  s->type = (int)transfo;
  s->n = (transfo == PFFFT_REAL) ? N / 2 : N;
  if (s->n < 1) { free(s); return NULL; }
  const size_t bytes = sizeof(float) * 4u * (unsigned)(2u * (unsigned)s->n) + 8u;
  s->sc = (float*)malloc(bytes);
  return s;
}

void pffft_destroy_setup(PFFFT_Setup_data* s) {
  if (!s) return;
  free(s->sc);
  free(s);
}

static void real_fwd(const PFFFT_Setup_data* s, const float* in, float* out) {
  const int n = s->n;
  float* plane = s->sc;
  float* out_c = s->sc + 2u * (unsigned)n * 2u; /* second plane */
  for (int t = 0; t < n; ++t) {
    plane[2 * t] = in[2 * t];
    plane[2 * t + 1] = in[2 * t + 1];
  }
  cfft_fwd(plane, out_c, s->sc + 4u * (unsigned)n, n);
  for (int k = 1; k < n; ++k) {
    const double ang = -PI * k / n;
    const double c = cos(ang), sn = sin(ang);
    const double A_r = 0.5 * (out_c[2 * k] + out_c[2 * (n - k)]);
    const double A_i = 0.5 * (out_c[2 * k + 1] - out_c[2 * (n - k) + 1]);
    const double B_r = 0.5 * (out_c[2 * k + 1] + out_c[2 * (n - k) + 1]);
    const double B_i = 0.5 * (out_c[2 * k] - out_c[2 * (n - k)]);
    const double Xr = A_r + (c * B_r - sn * B_i);
    const double Xi = A_i + (c * B_i + sn * B_r);
    out[2 + 2 * (k - 1)] = (float)Xr;
    out[3 + 2 * (k - 1)] = (float)Xi;
  }
  const double A0r = 0.5 * (out_c[0] + out_c[1]);
  const double A0i = 0.5 * (out_c[1] - out_c[0]);
  out[0] = (float)(A0r + A0i);
  out[1] = (float)(A0r - A0i);
}

static void real_inv(const PFFFT_Setup_data* s, const float* in, float* out) {
  const int n = s->n;
  float* plane = s->sc;
  float* Yc = s->sc + 4u * (unsigned)n; /* second plane */
  for (int k = 1; k < n; ++k) {
    const double ang = PI * k / n;
    const double c = cos(ang), sn = sin(ang);
    const double Xr = in[2 + 2 * (k - 1)];
    const double Xi = in[3 + 2 * (k - 1)];
    const double Xmr = in[2 * (n - k) > 1 ? 2 + 2 * (n - k - 1) : 0];
    const double Xmi = in[2 * (n - k) > 1 ? 3 + 2 * (n - k - 1) : 1];
    const double A_r = 0.5 * (Xr + Xmr);
    const double A_i = 0.5 * (Xi - Xmi);
    const double B_r = 0.5 * (Xi + Xmi);
    const double B_i = 0.5 * (Xr - Xmr);
    const double Yr = A_r + (c * B_r - sn * B_i);
    const double Yi = A_i + (c * B_i + sn * B_r);
    Yc[2 * k] = (float)Yr;
    Yc[2 * k + 1] = (float)Yi;
  }
  const double Zr = in[0], Zi = in[1];
  Yc[0] = (float)(0.5 * (Zr + Zi));
  Yc[1] = (float)(0.5 * (Zr - Zi));
  Yc[2 * n - 2] = (float)Yc[0]; /* index n: k=n shares with 0; E[n]=E[0] */
  Yc[2 * n - 1] = (float)Yc[1];
  {
    const size_t bytes = sizeof(float) * 2u * (unsigned)n;
    float* tmp = (float*)malloc(bytes);
    memcpy(tmp, Yc, bytes);
    cfft_inv(tmp, plane, s->sc + 2u * (unsigned)n * 2u + 2u * (unsigned)n, n);
    for (int t = 0; t < n; ++t) {
      out[2 * t] = plane[2 * t] * (float)(2 * n);
      out[2 * t + 1] = plane[2 * t + 1] * (float)(2 * n);
    }
    free(tmp);
  }
}

void pffft_transform(PFFFT_Setup_data* s, const float* in, float* out,
                     float* work, int direction) {
  pffft_transform_ordered(s, in, out, work, direction);
}

void pffft_transform_ordered(PFFFT_Setup_data* s, const float* in, float* out,
                             float* work, int direction) {
  (void)work;
  if (s->type == PFFFT_REAL) {
    if (direction == PFFFT_FORWARD) real_fwd(s, in, out);
    else real_inv(s, in, out);
    return;
  }
  const int n = s->n;
  if (direction == PFFFT_FORWARD) cfft_fwd(in, out, s->sc, n);
  else cfft_inv(in, out, s->sc, n);
}

void pffft_zconvolve_accumulate(PFFFT_Setup_data* s, const float* A,
                                const float* B, float* C) {
  if (s->type == PFFFT_COMPLEX) {
    const int n = s->n;
    for (int k = 0; k < n; ++k) {
      const double ar = A[2 * k], ai = A[2 * k + 1];
      const double br = B[2 * k], bi = B[2 * k + 1];
      C[2 * k] += (float)(ar * br - ai * bi);
      C[2 * k + 1] += (float)(ar * bi + ai * br);
    }
    return;
  }
  const int n = s->n;
  C[0] += A[0] * B[0];
  C[1] += A[1] * B[1];
  for (int k = 1; k < n; ++k) {
    const double ar = A[2 + 2 * (k - 1)], ai = A[3 + 2 * (k - 1)];
    const double br = B[2 + 2 * (k - 1)], bi = B[3 + 2 * (k - 1)];
    C[2 + 2 * (k - 1)] += (float)(ar * br - ai * bi);
    C[3 + 2 * (k - 1)] += (float)(ar * bi + ai * br);
  }
}
