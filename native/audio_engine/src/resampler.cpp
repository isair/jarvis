#include "resampler.h"

#include <cmath>
#include <cstring>

namespace {
constexpr double kPi = 3.14159265358979323846;

/* Windowed-sinc value, Hamming window. `c` = normalized cutoff (1 for
 * 1x, 1/3 for the 48k->16k decimator), `t` = tap offset from centre. */
inline double SincTap(double t, double c) {
  const double x = t * c;
  const double s = (std::fabs(x) < 1e-12) ? 1.0 : std::sin(kPi * x) / (kPi * x);
  return s;
}
}  // namespace

/* ---------------------------- Resampler16k ------------------------------ */

Resampler16k::Resampler16k() : hist_len_(0), abs_(0) {
  hist_len_ = 0;
  std::memset(hist_, 0, sizeof(hist_));
  /* Hamming-windowed sinc, cutoff at 1/3 (normalized to input rate). */
  const int n = kTaps;
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    const double t = i - (n - 1) / 2.0;
    const double s = SincTap(t / 3.0, 1.0);
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

/* --------------------------- ClockedResampler --------------------------- */

ClockedResampler::ClockedResampler() {
  /* Build the kPhases x kTaps polyphase bank at cutoff = min(1, step/1)
   * recomputed per configure(); the table below is for step == 1 and is
   * rebuilt in configure(). */
  std::memset(table_, 0, sizeof(table_));
  configure(48000u, 48000u);
}

void ClockedResampler::configure(uint32_t in_rate_hz, uint32_t out_rate_hz) {
  if (in_rate_hz == 0u || out_rate_hz == 0u) in_rate_hz = out_rate_hz = 48000u;
  step_ = static_cast<double>(in_rate_hz) / static_cast<double>(out_rate_hz);
  pos_ = 0.0;
  /* Cutoff so the wider of the two Nyquist bands passes: when decimating
   * (step > 1) the cutoff shrinks to 1/step of the input rate. */
  const double c = (step_ >= 1.0) ? (1.0 / step_) : 1.0;
  /* Normalized so the DC gain is exactly 1 after the per-phase sum. */
  double g[2];
  g[0] = 0.0;
  g[1] = 0.0;
  for (int p = 0; p < kPhases; ++p) {
    const double phase = static_cast<double>(p) / static_cast<double>(kPhases);
    const double centre = static_cast<double>(kTaps - 1) * 0.5;
    double ssum = 0.0;
    for (int i = 0; i < kTaps; ++i) {
      const double t = static_cast<double>(i) - centre + (0.5 - phase);
      const double w = 0.54 + 0.46 * std::cos(2.0 * kPi * i / (kTaps - 1));
      const double v = SincTap(t * c, 1.0) * w;
      table_[p][i] = static_cast<float>(v);
      ssum += v;
    }
    g[p % 2 == 0 ? 0 : 1] += ssum;
  }
  /* Equalize the phase-group sums so all phases share one gain. */
  const double mean_g = (g[0] / (kPhases / 2) + g[1] / (kPhases / 2)) * 0.5;
  if (mean_g != 0.0) {
    for (int p = 0; p < kPhases; ++p) {
      double sum = 0.0;
      for (int i = 0; i < kTaps; ++i) sum += table_[p][i];
      if (sum == 0.0) continue;
      const double scale = mean_g / sum;
      for (int i = 0; i < kTaps; ++i) table_[p][i] = static_cast<float>(table_[p][i] * scale);
    }
  }
}

void ClockedResampler::reset() { pos_ = 0.0; }

void ClockedResampler::set_ppm(double ppm) {
  ppm_ = ppm;
  if (ppm_ > 1000.0) ppm_ = 1000.0;
  if (ppm_ < -1000.0) ppm_ = -1000.0;
}

size_t ClockedResampler::process(const float* in, size_t n_in, float* out,
                                 size_t out_cap) {
  if (n_in == 0) return 0;
  /* Effective step: the ppm correction scales the output clock, i.e. it
   * shrinks/grows the input samples consumed per produced output sample. */
  const double step_eff = step_ * (1.0 + ppm_ * 1e-6);
  const int half = kTaps / 2;  /* taps/2 support on each side of the centre */
  size_t produced = 0;
  /* `pos_` is the centre position of the next output, in this call's input
   * coordinates. Align it into the supported range once, then advance by the
   * continuous (possibly ppm-corrected) phase increment. */
  while (pos_ < static_cast<double>(half)) pos_ += step_eff;
  while (pos_ <= static_cast<double>(n_in) + half) {
    const double from = pos_ - half;
    const int base_idx = static_cast<int>(std::floor(from));
    const double frac = from - static_cast<double>(base_idx);
    int phase = static_cast<int>(frac * kPhases + 0.5);
    if (phase >= kPhases) phase = 0;
    float acc = 0.0f;
    const float* h = table_[phase];
    const int first = base_idx < 0 ? 0 : base_idx;
    const int last =
        (base_idx + kTaps > static_cast<int>(n_in)) ? static_cast<int>(n_in)
                                                    : base_idx + kTaps;
    for (int i = first; i < last; ++i) acc += h[i - base_idx] * in[i];
    out[produced++] = acc;
    pos_ += step_eff;
    if (produced >= out_cap) break;
  }
  /* Re-origin for the next call: coordinates shift by the consumed input. */
  pos_ -= static_cast<double>(n_in);
  return produced;
}
