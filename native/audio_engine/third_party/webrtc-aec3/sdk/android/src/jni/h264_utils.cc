/*
 *  Copyright 2018 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "api/video_codecs/h264_profile_level_id.h"
#include "sdk/android/generated_video_jni/H264Utils_jni.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/video_codec_info.h"
#include "third_party/jni_zero/jni_zero.h"

namespace webrtc {
namespace jni {

static jboolean JNI_H264Utils_IsSameH264Profile(
    JNIEnv* env,
    const jni_zero::JavaParamRef<jobject>& params1,
    const jni_zero::JavaParamRef<jobject>& params2) {
  return H264IsSameProfile(JavaToNativeStringMap(env, params1),
                           JavaToNativeStringMap(env, params2));
}

}  // namespace jni
}  // namespace webrtc
