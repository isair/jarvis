/*
 *  Copyright 2025 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include "api/environment/environment.h"

#include <jni.h>

#include <memory>
#include <utility>

#include "api/environment/environment_factory.h"
#include "api/field_trials.h"
#include "api/field_trials_view.h"
#include "sdk/android/generated_environment_jni/Environment_jni.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "third_party/jni_zero/jni_zero.h"

namespace webrtc::jni {

void JNI_Environment_Free(JNIEnv* /*env*/, jlong webrtcEnv) {
  delete reinterpret_cast<Environment*>(webrtcEnv);
}

jlong JNI_Environment_Create(
    JNIEnv* env,
    const jni_zero::JavaParamRef<jstring>& fieldTrials) {
  std::unique_ptr<FieldTrialsView> field_trials;
  if (!IsNull(env, fieldTrials)) {
    field_trials =
        std::make_unique<FieldTrials>(JavaToNativeString(env, fieldTrials));
  }
  return NativeToJavaPointer(
      new Environment(CreateEnvironment(std::move(field_trials))));
}

}  // namespace webrtc::jni
