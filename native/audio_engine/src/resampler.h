/* resampler.h — band-limited polyphase 48k -> 16k (integer ratio 3). */
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

#endif /* JARVIS_RESAMPLER_H_ */
