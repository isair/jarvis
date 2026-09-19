/*
 * Copyright (c) 2013 The WebRTC project authors. All Rights Reserved.
 *
 * Use of this source code is governed by a BSD-style license
 * that can be found in the LICENSE file in the root of the source
 * tree. An additional intellectual property rights grant can be found
 * in the file PATENTS.  All contributing project authors may
 * be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VIDEO_CONFIG_VIDEO_ENCODER_CONFIG_H_
#define VIDEO_CONFIG_VIDEO_ENCODER_CONFIG_H_

#include <stddef.h>

#include <optional>
#include <string>
#include <vector>

#include "api/field_trials_view.h"
#include "api/ref_count.h"
#include "api/scoped_refptr.h"
#include "api/video/resolution.h"
#include "api/video/video_codec_type.h"
#include "api/video_codecs/scalability_mode.h"
#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/spatial_layer.h"
#include "api/video_codecs/video_codec.h"

namespace webrtc {

// The `VideoStream` struct describes a simulcast layer, or "stream".
struct VideoStream {
  VideoStream();
  ~VideoStream();
  VideoStream(const VideoStream& other);
  std::string ToString() const;

  // Width/Height in pixels.
  size_t width;
  size_t height;

  // Frame rate in fps.
  int max_framerate;

  // Bitrate, in bps, for the stream.
  int min_bitrate_bps;
  int target_bitrate_bps;
  int max_bitrate_bps;

  // Scaling factor applied to the stream size.
  double scale_resolution_down_by;

  // Maximum Quantization Parameter to use when encoding the stream.
  int max_qp;

  std::optional<size_t> num_temporal_layers;
  std::optional<double> bitrate_priority;
  std::optional<ScalabilityMode> scalability_mode;

  // If this stream is enabled by the user, or not.
  bool active;

  std::optional<Resolution> scale_resolution_down_to;
};

class VideoEncoderConfig {
 public:
  class EncoderSpecificSettings : public RefCountInterface {
   public:
    void FillEncoderSpecificSettings(VideoCodec* codec_struct) const;

    virtual void FillVideoCodecVp8(VideoCodecVP8* vp8_settings) const;
    virtual void FillVideoCodecVp9(VideoCodecVP9* vp9_settings) const;
    virtual void FillVideoCodecAv1(VideoCodecAV1* av1_settings) const;

   private:
    ~EncoderSpecificSettings() override {}
    friend class VideoEncoderConfig;
  };

  class Vp8EncoderSpecificSettings : public EncoderSpecificSettings {
   public:
    explicit Vp8EncoderSpecificSettings(const VideoCodecVP8& specifics);
    void FillVideoCodecVp8(VideoCodecVP8* vp8_settings) const override;

   private:
    VideoCodecVP8 specifics_;
  };

  class Vp9EncoderSpecificSettings : public EncoderSpecificSettings {
   public:
    explicit Vp9EncoderSpecificSettings(const VideoCodecVP9& specifics);
    void FillVideoCodecVp9(VideoCodecVP9* vp9_settings) const override;

   private:
    VideoCodecVP9 specifics_;
  };

  class Av1EncoderSpecificSettings : public EncoderSpecificSettings {
   public:
    explicit Av1EncoderSpecificSettings(const VideoCodecAV1& specifics);
    void FillVideoCodecAv1(VideoCodecAV1* av1_settings) const override;

   private:
    VideoCodecAV1 specifics_;
  };

  enum class ContentType {
    kRealtimeVideo,
    kScreen,
  };

  class VideoStreamFactoryInterface : public RefCountInterface {
   public:
    virtual std::vector<VideoStream> CreateEncoderStreams(
        const FieldTrialsView& field_trials,
        int frame_width,
        int frame_height,
        const VideoEncoderConfig& encoder_config) = 0;

   protected:
    ~VideoStreamFactoryInterface() override {}
  };

  VideoEncoderConfig& operator=(VideoEncoderConfig&&) = default;
  VideoEncoderConfig& operator=(const VideoEncoderConfig&) = delete;

  // Mostly used by tests. Avoid creating copies if you can.
  VideoEncoderConfig Copy() const { return VideoEncoderConfig(*this); }

  VideoEncoderConfig();
  VideoEncoderConfig(VideoEncoderConfig&&);
  ~VideoEncoderConfig();
  std::string ToString() const;

  bool HasScaleResolutionDownTo() const;

  VideoCodecType codec_type;
  SdpVideoFormat video_format;

  scoped_refptr<VideoStreamFactoryInterface> video_stream_factory;
  std::vector<SpatialLayer> spatial_layers;
  ContentType content_type;
  bool frame_drop_enabled;
  scoped_refptr<const EncoderSpecificSettings> encoder_specific_settings;

  int min_transmit_bitrate_bps;
  int max_bitrate_bps;
  double bitrate_priority;

  std::vector<VideoStream> simulcast_layers;

  size_t number_of_streams;

  // Legacy Google conference mode flag for simulcast screenshare
  bool legacy_conference_mode;

  bool is_quality_scaling_allowed;

  int max_qp;

 private:
  VideoEncoderConfig(const VideoEncoderConfig&);
};

}  // namespace webrtc

#endif  // VIDEO_CONFIG_VIDEO_ENCODER_CONFIG_H_
