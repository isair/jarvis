/* resampler.h — band-limited polyphase conversions for the AEC/ASR domain.
 *
 * Resampler16k  : exact integer ratio 48k -> 16k (decimate by 3, 1:3).
 * ClockedResampler : arbitrary source -> 48 kHz with a continuously
 *                    adjustable phase increment. A PI controller on the
 *                    queue/timeline error drives the ppm correction
 *                    (clamped to +/-1000 ppm in the engine). Discontinuities
 *                    use an explicit hard reset, never per-sample drops.
 */
#ifndef JARVIS_RESAMPLER_H_
#define JARVIS_RESAMPLER_H_

#include <cstddef>
#include <cstdint>

/* Windowed-sinc lowpass, 24 taps @ 48 kHz, cutoff = 1/3 of input rate
 * (Nyquist of the 16 kHz output). Integer-ratio decimation by 3:
 *   y(k) = sum_{i=0..23} h[i] * x(3k + i - 23)
 * No linear interpolation; true band-limited polyphase form. */
class Resampler16k {
 public:
  Resampler16k();
  void reset();
  /* n_in: input samples (any length; leftover is kept in history). */
  size_t process(const float* in, size_t n_in, float* out, size_t out_cap);

 private:
  static constexpr int kTaps = 24;
  static constexpr int kRatio = 3;
  float h_[kTaps];
  float hist_[kTaps];   /* last kTaps-1 inputs, newest at index kTaps-2 */
  size_t hist_len_;
  size_t abs_;  /* absolute input sample count fed so far (phase tracking) */
};

/* Fractional-ratio polyphase resampler toward a nominal 48 kHz output.
 *
 * Output n samples the input at fractional position pos = n * step, where
 *   step = (in_rate / 48000) / (1 + ppm * 1e-6)
 * and `ppm` is the signed clock-drift correction from the lane PI
 * controller. 8-phase x 16-tap windowed-sinc bank; phases between table
 * entries are linearized (phase resolution 1/8 sample, 125 ns at 48 kHz).
 */
class ClockedResampler {
 public:
  ClockedResampler();
  void configure(uint32_t in_rate_hz, uint32_t out_rate_hz = 48000u);
  void reset();
  /* Signed drift correction, parts per million (already clamped). */
  void set_ppm(double ppm);
  double ppm() const { return ppm_; }
  double step() const { return step_; }
  /* Resample `n_in` samples; produced count <= out_cap. */
  size_t process(const float* in, size_t n_in, float* out, size_t out_cap);

 private:
  static constexpr int kTaps = 16;   /* even, windowed sinc pair-sums */
  static constexpr int kPhases = 16; /* table phases per input sample */
  double step_ = 1.0;                /* input samples per output sample */
  double pos_ = 0.0;                 /* fractional position in the input */
  double ppm_ = 0.0;
  float table_[kPhases][kTaps];
};

#endif /* JARVIS_RESAMPLER_H_ */
