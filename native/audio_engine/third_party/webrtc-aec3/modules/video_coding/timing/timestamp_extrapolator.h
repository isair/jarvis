/*
 *  Copyright (c) 2011 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef MODULES_VIDEO_CODING_TIMING_TIMESTAMP_EXTRAPOLATOR_H_
#define MODULES_VIDEO_CODING_TIMING_TIMESTAMP_EXTRAPOLATOR_H_

#include <stdint.h>

#include <memory>
#include <optional>

#include "absl/strings/string_view.h"
#include "api/field_trials_view.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "rtc_base/experiments/struct_parameters_parser.h"
#include "rtc_base/numerics/sequence_number_unwrapper.h"

namespace webrtc {

// Not thread safe.
class TimestampExtrapolator {
 public:
  // Configuration struct for overriding some constants and behaviour,
  // configurable through field trials.
  struct Config {
    inline static constexpr absl::string_view kFieldTrialsKey =
        "WebRTC-TimestampExtrapolatorConfig";

    // Factory function that parses the field trials and returns a `Config`
    // with validated values.
    static Config ParseAndValidate(const FieldTrialsView& field_trials);

    std::unique_ptr<StructParametersParser> Parser() {
      // clang-format off
      return StructParametersParser::Create(
        "hard_reset_timeout", &hard_reset_timeout,
        "alarm_threshold", &alarm_threshold,
        "acc_drift", &acc_drift,
        "acc_max_error", &acc_max_error,
        "reset_full_cov_on_alarm", &reset_full_cov_on_alarm);
      // clang-format on
    }

    // If a frame has not been received within this timeout, do a full reset.
    TimeDelta hard_reset_timeout = TimeDelta::Seconds(10);

    // Alarm on sudden delay change if the (filtered; see below) accumulated
    // residuals are larger than this number of RTP ticks. After the
    // startup period, an alarm will result in a full or partial reset of the
    // uncertainty covariance (see `reset_full_cov_on_alarm` below).
    int alarm_threshold = 60000;  // 666 ms <=> 20 frames@30fps.

    // Acceptable level of per-frame drift in the detector (in RTP ticks).
    int acc_drift = 6600;  // 73 ms <=> 2.2 frames@30fps.

    // Max limit on residuals in the detector (in RTP ticks).
    // TODO(brandtr): Increase from this unreasonably low value.
    int acc_max_error = 7000;  // 77 ms <=> 2.3 frames@30fps.

    // If true, reset the entire uncertainty covariance matrix on alarms.
    // If false, only reset the offset variance term.
    // TODO(brandtr): Flip so that the frequency term won't get hit tpp badly
    // when a large delay spike happens.
    bool reset_full_cov_on_alarm = false;
  };

  TimestampExtrapolator(Timestamp start, const FieldTrialsView& field_trials);
  ~TimestampExtrapolator();
  void Update(Timestamp now, uint32_t ts90khz);
  std::optional<Timestamp> ExtrapolateLocalTime(uint32_t timestamp90khz) const;
  void Reset(Timestamp start);

  Config GetConfigForTest() const { return config_; }

 private:
  void CheckForWrapArounds(uint32_t ts90khz);
  bool DelayChangeDetection(double error);

  const Config config_;

  double w_[2];
  double p_[2][2];
  Timestamp start_;
  Timestamp prev_;
  std::optional<int64_t> first_unwrapped_timestamp_;
  RtpTimestampUnwrapper unwrapper_;
  std::optional<int64_t> prev_unwrapped_timestamp_;
  int packet_count_;
  double detector_accumulator_pos_;
  double detector_accumulator_neg_;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_TIMING_TIMESTAMP_EXTRAPOLATOR_H_
