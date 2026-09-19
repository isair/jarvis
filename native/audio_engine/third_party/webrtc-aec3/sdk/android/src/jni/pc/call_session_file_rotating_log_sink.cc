/*
 *  Copyright 2017 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "rtc_base/log_sinks.h"
#include "sdk/android/generated_peerconnection_jni/CallSessionFileRotatingLogSink_jni.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/jni_helpers.h"
#include "third_party/jni_zero/jni_zero.h"

namespace webrtc {
namespace jni {

static jlong JNI_CallSessionFileRotatingLogSink_AddSink(
    JNIEnv* jni,
    const jni_zero::JavaParamRef<jstring>& j_dirPath,
    jint j_maxFileSize,
    jint j_severity) {
  std::string dir_path = JavaToStdString(jni, j_dirPath);
  CallSessionFileRotatingLogSink* sink =
      new CallSessionFileRotatingLogSink(dir_path, j_maxFileSize);
  if (!sink->Init()) {
    RTC_LOG_V(LoggingSeverity::LS_WARNING)
        << "Failed to init CallSessionFileRotatingLogSink for path "
        << dir_path;
    delete sink;
    return 0;
  }
  LogMessage::AddLogToStream(sink, static_cast<LoggingSeverity>(j_severity));
  return jlongFromPointer(sink);
}

static void JNI_CallSessionFileRotatingLogSink_DeleteSink(JNIEnv* jni,
                                                          jlong j_sink) {
  CallSessionFileRotatingLogSink* sink =
      reinterpret_cast<CallSessionFileRotatingLogSink*>(j_sink);
  LogMessage::RemoveLogToStream(sink);
  delete sink;
}

static jni_zero::ScopedJavaLocalRef<jbyteArray>
JNI_CallSessionFileRotatingLogSink_GetLogData(
    JNIEnv* jni,
    const jni_zero::JavaParamRef<jstring>& j_dirPath) {
  std::string dir_path = JavaToStdString(jni, j_dirPath);
  CallSessionFileRotatingStreamReader file_reader(dir_path);
  size_t log_size = file_reader.GetSize();
  if (log_size == 0) {
    RTC_LOG_V(LoggingSeverity::LS_WARNING)
        << "CallSessionFileRotatingStream returns 0 size for path " << dir_path;
    return jni_zero::ScopedJavaLocalRef<jbyteArray>(jni, jni->NewByteArray(0));
  }

  // TODO(nisse, sakal): To avoid copying, change api to use ByteBuffer.
  std::unique_ptr<jbyte> buffer(static_cast<jbyte*>(malloc(log_size)));
  size_t read = file_reader.ReadAll(buffer.get(), log_size);

  jni_zero::ScopedJavaLocalRef<jbyteArray> result =
      jni_zero::ScopedJavaLocalRef<jbyteArray>(jni, jni->NewByteArray(read));
  jni->SetByteArrayRegion(result.obj(), 0, read, buffer.get());

  return result;
}

}  // namespace jni
}  // namespace webrtc
