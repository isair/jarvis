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

#include "api/environment/environment.h"
#include "modules/video_coding/codecs/vp8/include/vp8.h"
#include "sdk/android/generated_libvpx_vp8_jni/LibvpxVp8Decoder_jni.h"
#include "sdk/android/generated_libvpx_vp8_jni/LibvpxVp8Encoder_jni.h"
#include "sdk/android/native_api/jni/java_types.h"

namespace webrtc {
namespace jni {

jlong JNI_LibvpxVp8Encoder_Create(JNIEnv* jni, jlong j_webrtc_env_ref) {
  return NativeToJavaPointer(
      CreateVp8Encoder(*reinterpret_cast<const Environment*>(j_webrtc_env_ref))
          .release());
}

static jlong JNI_LibvpxVp8Decoder_CreateDecoder(JNIEnv* jni,
                                                jlong j_webrtc_env_ref) {
  return NativeToJavaPointer(
      CreateVp8Decoder(*reinterpret_cast<const Environment*>(j_webrtc_env_ref))
          .release());
}

}  // namespace jni
}  // namespace webrtc
