/*
 *  Copyright 2018 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "sdk/android/src/jni/pc/rtc_certificate.h"

#include "rtc_base/ref_count.h"
#include "rtc_base/rtc_certificate.h"
#include "rtc_base/rtc_certificate_generator.h"
#include "sdk/android/generated_peerconnection_jni/RtcCertificatePem_jni.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/jni_helpers.h"
#include "sdk/android/src/jni/pc/ice_candidate.h"
#include "third_party/jni_zero/jni_zero.h"

namespace webrtc {
namespace jni {

RTCCertificatePEM JavaToNativeRTCCertificatePEM(
    JNIEnv* jni,
    const jni_zero::JavaRef<jobject>& j_rtc_certificate) {
  jni_zero::ScopedJavaLocalRef<jstring> privatekey_field =
      Java_RtcCertificatePem_getPrivateKey(jni, j_rtc_certificate);
  jni_zero::ScopedJavaLocalRef<jstring> certificate_field =
      Java_RtcCertificatePem_getCertificate(jni, j_rtc_certificate);
  return RTCCertificatePEM(JavaToNativeString(jni, privatekey_field),
                           JavaToNativeString(jni, certificate_field));
}

ScopedJavaLocalRef<jobject> NativeToJavaRTCCertificatePEM(
    JNIEnv* jni,
    const RTCCertificatePEM& certificate) {
  return Java_RtcCertificatePem_Constructor(
      jni, NativeToJavaString(jni, certificate.private_key()),
      NativeToJavaString(jni, certificate.certificate()));
}

static jni_zero::ScopedJavaLocalRef<jobject>
JNI_RtcCertificatePem_GenerateCertificate(
    JNIEnv* jni,
    const jni_zero::JavaParamRef<jobject>& j_key_type,
    jlong j_expires) {
  KeyType key_type = JavaToNativeKeyType(jni, j_key_type);
  uint64_t expires = (uint64_t)j_expires;
  scoped_refptr<RTCCertificate> certificate =
      RTCCertificateGenerator::GenerateCertificate(KeyParams(key_type),
                                                   expires);
  RTCCertificatePEM pem = certificate->ToPEM();
  return Java_RtcCertificatePem_Constructor(
      jni, NativeToJavaString(jni, pem.private_key()),
      NativeToJavaString(jni, pem.certificate()));
}

}  // namespace jni
}  // namespace webrtc
