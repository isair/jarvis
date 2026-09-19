/*
 *  Copyright 2018 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include "sdk/android/native_api/peerconnection/peer_connection_factory.h"

#include <jni.h>

#include <cstddef>
#include <memory>
#include <utility>

#include "api/enable_media_with_defaults.h"
#include "api/environment/environment_factory.h"
#include "api/peer_connection_interface.h"
#include "api/rtc_event_log/rtc_event_log_factory.h"
#include "api/scoped_refptr.h"
#include "media/engine/internal_decoder_factory.h"
#include "media/engine/internal_encoder_factory.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "rtc_base/physical_socket_server.h"
#include "rtc_base/thread.h"
#include "sdk/android/generated_native_unittests_jni/PeerConnectionFactoryInitializationHelper_jni.h"
#include "sdk/android/native_api/audio_device_module/audio_device_android.h"
#include "sdk/android/native_api/jni/application_context_provider.h"
#include "sdk/android/native_api/jni/jvm.h"
#include "test/gtest.h"

namespace webrtc {
namespace test {
namespace {

// Create native peer connection factory, that will be wrapped by java one
webrtc::scoped_refptr<webrtc::PeerConnectionFactoryInterface> CreateTestPCF(
    JNIEnv* jni,
    webrtc::Thread* network_thread,
    webrtc::Thread* worker_thread,
    webrtc::Thread* signaling_thread) {
  // talk/ assumes pretty widely that the current Thread is ThreadManager'd, but
  // ThreadManager only WrapCurrentThread()s the thread where it is first
  // created.  Since the semantics around when auto-wrapping happens in
  // webrtc/rtc_base/ are convoluted, we simply wrap here to avoid having to
  // think about ramifications of auto-wrapping there.
  webrtc::ThreadManager::Instance()->WrapCurrentThread();

  PeerConnectionFactoryDependencies pcf_deps;
  pcf_deps.network_thread = network_thread;
  pcf_deps.worker_thread = worker_thread;
  pcf_deps.signaling_thread = signaling_thread;
  pcf_deps.event_log_factory = std::make_unique<RtcEventLogFactory>();
  pcf_deps.env = CreateEnvironment();

  pcf_deps.adm =
      CreateJavaAudioDeviceModule(jni, *pcf_deps.env, GetAppContext(jni).obj());
  pcf_deps.video_encoder_factory =
      std::make_unique<webrtc::InternalEncoderFactory>();
  pcf_deps.video_decoder_factory =
      std::make_unique<webrtc::InternalDecoderFactory>();
  EnableMediaWithDefaults(pcf_deps);

  auto factory = CreateModularPeerConnectionFactory(std::move(pcf_deps));
  RTC_LOG(LS_INFO) << "PeerConnectionFactory created: " << factory.get();
  RTC_CHECK(factory) << "Failed to create the peer connection factory; "
                        "WebRTC/libjingle init likely failed on this device";

  return factory;
}

TEST(PeerConnectionFactoryTest, NativeToJavaPeerConnectionFactory) {
  JNIEnv* jni = AttachCurrentThreadIfNeeded();

  RTC_LOG(LS_INFO) << "Initializing java peer connection factory.";
  jni::Java_PeerConnectionFactoryInitializationHelper_initializeFactoryForTests(
      jni);
  RTC_LOG(LS_INFO) << "Java peer connection factory initialized.";

  auto socket_server = std::make_unique<webrtc::PhysicalSocketServer>();

  // Create threads.
  auto network_thread = std::make_unique<webrtc::Thread>(socket_server.get());
  network_thread->SetName("network_thread", nullptr);
  RTC_CHECK(network_thread->Start()) << "Failed to start thread";

  std::unique_ptr<webrtc::Thread> worker_thread = webrtc::Thread::Create();
  worker_thread->SetName("worker_thread", nullptr);
  RTC_CHECK(worker_thread->Start()) << "Failed to start thread";

  std::unique_ptr<webrtc::Thread> signaling_thread = webrtc::Thread::Create();
  signaling_thread->SetName("signaling_thread", NULL);
  RTC_CHECK(signaling_thread->Start()) << "Failed to start thread";

  webrtc::scoped_refptr<webrtc::PeerConnectionFactoryInterface> factory =
      CreateTestPCF(jni, network_thread.get(), worker_thread.get(),
                    signaling_thread.get());

  jobject java_factory = NativeToJavaPeerConnectionFactory(
      jni, factory, std::move(socket_server), std::move(network_thread),
      std::move(worker_thread), std::move(signaling_thread));

  RTC_LOG(LS_INFO) << java_factory;

  EXPECT_NE(java_factory, nullptr);
}

}  // namespace
}  // namespace test
}  // namespace webrtc
