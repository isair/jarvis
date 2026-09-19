/*
 *  Copyright (c) 2018 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <jni.h>
#undef JNIEXPORT
#define JNIEXPORT __attribute__((visibility("default")))

#include "rtc_base/logging.h"
#include "sdk/android/native_api/base/init.h"
#include "test/android/native_test_launcher.h"  // nogncheck
#include "third_party/jni_zero/jni_zero.h"

// This is called by the VM when the shared library is first loaded.
JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
  RTC_LOG(LS_INFO) << "Entering JNI_OnLoad in test_jni_onload.cc";

  jni_zero::InitVM(vm);
  // TODO(webrtc:42223878): Set exception handler?
  // jni_zero::SetExceptionHandler(CheckException);
  // JNIEnv* env = jni_zero::AttachCurrentThread();
  // TODO(webrtc:42223878): Classloader, OOM error handler?
  webrtc::InitAndroid(vm);
  webrtc::test::android::InstallHandlers();
  return JNI_VERSION_1_4;
}
