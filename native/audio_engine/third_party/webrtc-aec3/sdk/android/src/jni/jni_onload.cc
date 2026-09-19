/*
 *  Copyright 2015 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <jni.h>

#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "rtc_base/ssl_adapter.h"
#include "sdk/android/native_api/jni/class_loader.h"
#include "sdk/android/src/jni/jvm.h"

#undef JNIEXPORT
#define JNIEXPORT __attribute__((visibility("default")))

namespace webrtc {
namespace jni {

extern "C" jint JNIEXPORT JNICALL JNI_OnLoad(JavaVM* jvm, void* reserved) {
  RTC_LOG(LS_INFO) << "Entering JNI_OnLoad in jni_onload.cc";
  jint ret = InitGlobalJniVariables(jvm);
  RTC_DCHECK_GE(ret, 0);
  if (ret < 0)
    return -1;

  RTC_CHECK(InitializeSSL()) << "Failed to InitializeSSL()";
  InitClassLoader(GetEnv());

  return ret;
}

extern "C" void JNIEXPORT JNICALL JNI_OnUnLoad(JavaVM* jvm, void* reserved) {
  RTC_CHECK(CleanupSSL()) << "Failed to CleanupSSL()";
}

}  // namespace jni
}  // namespace webrtc
