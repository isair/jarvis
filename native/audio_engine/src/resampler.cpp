#include "resampler.h"

#include <cmath>
#include <cstring>

namespace {
constexpr double kPi = 3.14159265358979323846;
}

Resampler16k::Resampler16k() : hist_len_(0), abs_(0) {
  hist_len_ = 0;
  std::memset(hist_, 0, sizeof(hist_));
  /* Hamming-windowed sinc, cutoff at 1/3 (normalized to input rate). */
  const int n = kTaps;
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    const double t = i - (n - 1) / 2.0;
    const double x = t / 3.0;  /* cutoff 1/3 */
    const double s = (std::fabs(x) < 1e-12) ? (1.0 / 3.0)
                                            : std::sin(kPi * x) / (kPi * x);
    const double w = 0.54 + 0.46 * std::cos(2.0 * kPi * i / (n - 1));
    h_[i] = static_cast<float>(s * w);
    sum += h_[i];
  }
  if (sum != 0.0)
    for (int i = 0; i < n; ++i) h_[i] = static_cast<float>(h_[i] / sum);
}

void Resampler16k::reset() {
  hist_len_ = 0;
  std::memset(hist_, 0, sizeof(hist_));
}

size_t Resampler16k::process(const float* in, size_t n_in, float* out,
                             size_t out_cap) {
  /* Join history and the new block into one logical stream. Merged position
   * j maps to absolute input index pos = abs_ - hist_len_ + j. Outputs exist
   * at absolute indices n with (n - (kTaps-1)) % kRatio == 0, so the phase
   * stays exact across arbitrary block sizes. */
  if (n_in == 0) return 0;
  const size_t hist_len = hist_len_;
  const size_t base = abs_ - hist_len;
  const size_t n_total = hist_len + n_in;
  size_t produced = 0;
  size_t idx = kTaps - 1u;
  if (idx < (base) + (kTaps - 1u) - base) {}
  while (idx < kTaps - 1u && idx < n_total) ++idx;
  const size_t n0 = base + idx;
  idx += (((kRatio - static_cast<int>((n0 - (kTaps - 1u)) % kRatio)) % kRatio));
  for (; idx < n_total; idx += kRatio) {
    float acc = 0.0f;
    for (int i = 0; i < kTaps; ++i) {
      const size_t pos = idx - (kTaps - 1) + i;
      const float x = (pos < hist_len) ? hist_[pos] : in[pos - hist_len];
      acc += h_[i] * x;
    }
    out[produced++] = acc;
    if (produced >= out_cap) break;
  }
  /* Keep the trailing kTaps-1 inputs as history. */
  const size_t tail = n_total > kTaps - 1u ? n_total - (kTaps - 1u) : 0u;
  size_t w = 0;
  for (size_t pos = tail; pos < n_total; ++pos)
    hist_[w++] = (pos < hist_len) ? hist_[pos] : in[pos - hist_len];
  hist_len_ = w;
  abs_ += n_in;
  return produced;
}
