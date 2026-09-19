/*
 *  Copyright 2017 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <jni.h>

#include <memory>
#include <utility>

#include "api/environment/environment.h"
#include "api/video_codecs/video_encoder.h"
#include "api/video_codecs/video_encoder_software_fallback_wrapper.h"
#include "sdk/android/generated_video_jni/VideoEncoderFallback_jni.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/video_encoder_wrapper.h"
#include "third_party/jni_zero/jni_zero.h"

namespace webrtc {
namespace jni {

jlong JNI_VideoEncoderFallback_Create(
    JNIEnv* jni,
    jlong j_webrtc_env_ref,
    const jni_zero::JavaParamRef<jobject>& j_fallback_encoder,
    const jni_zero::JavaParamRef<jobject>& j_primary_encoder) {
  std::unique_ptr<VideoEncoder> fallback_encoder =
      JavaToNativeVideoEncoder(jni, j_fallback_encoder, j_webrtc_env_ref);
  std::unique_ptr<VideoEncoder> primary_encoder =
      JavaToNativeVideoEncoder(jni, j_primary_encoder, j_webrtc_env_ref);

  return NativeToJavaPointer(
      CreateVideoEncoderSoftwareFallbackWrapper(
          *reinterpret_cast<const Environment*>(j_webrtc_env_ref),
          std::move(fallback_encoder), std::move(primary_encoder),
          /*prefer_temporal_support=*/false)
          .release());
}

}  // namespace jni
}  // namespace webrtc
