/* Minimal PFFFT-compatible implementation for the WebRTC audio processing
 * pipeline (ABI of the pffft API as used by modules/audio_processing/utility/
 * pffft_wrapper.cc). Public-domain style compact layout:
 *
 *  real transform (N reals):
 *    out[0]      = Re[X(0)]
 *    out[1]      = Re[X(N/2)]
 *    out[2+2k-2] = Re[X(k)],  out[3+2k-2] = Im[X(k)]   (k = 1..N/2-1)
 *  complex transform (N complex, 2N floats): plain interleaved DFT.
 *  backward = inverse with 1/N scale. Ordered and unordered produce the same
 *  compact spectrum per setup; both are self-consistent for one Pffft setup.
 */
#ifndef PFFFT_H_
#define PFFFT_H_

#ifdef __cplusplus
extern "C" {
#endif

enum PFFFT_1D { PFFFT_REAL = 1, PFFFT_COMPLEX = 2 };

#define PFFFT_FORWARD 0
#define PFFFT_BACKWARD 1

typedef struct PFFFT_Setup_data PFFFT_Setup_data;

void* pffft_aligned_malloc(size_t nb_bytes);
void pffft_aligned_free(void* p);

int pffft_simd_size(void);

PFFFT_Setup_data* pffft_new_setup(int N, PFFFT_1D transfo);
void pffft_destroy_setup(PFFFT_Setup_data*);

void pffft_transform(PFFFT_Setup_data*, const float* in, float* out,
                     float* work, int direction);
void pffft_transform_ordered(PFFFT_Setup_data*, const float* in, float* out,
                             float* work, int direction);
void pffft_zconvolve_accumulate(PFFFT_Setup_data*, const float* A,
                                const float* B, float* C);

#ifdef __cplusplus
}
#endif

#endif /* PFFFT_H_ */
