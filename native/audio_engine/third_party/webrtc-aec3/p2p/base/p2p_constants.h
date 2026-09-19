/*
 *  Copyright 2004 The WebRTC Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef P2P_BASE_P2P_CONSTANTS_H_
#define P2P_BASE_P2P_CONSTANTS_H_

#include <cstddef>
#include <cstdint>

#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// CN_ == "content name".  When we initiate a session, we choose the
// name, and when we receive a Gingle session, we provide default
// names (since Gingle has no content names). But when we receive a
// Jingle call, the content name can be anything, so don't rely on
// these values being the same as the ones received.
inline constexpr char CN_AUDIO[] = "audio";
inline constexpr char CN_VIDEO[] = "video";
inline constexpr char CN_DATA[] = "data";
inline constexpr char CN_OTHER[] = "main";

// GN stands for group name
inline constexpr char GROUP_TYPE_BUNDLE[] = "BUNDLE";

// Minimum ufrag length is 4 characters as per RFC5245.
inline constexpr int ICE_UFRAG_LENGTH = 4;
// Minimum password length of 22 characters as per RFC5245. We chose 24 because
// some internal systems expect password to be multiple of 4.
inline constexpr int ICE_PWD_LENGTH = 24;
inline constexpr size_t ICE_UFRAG_MIN_LENGTH = 4;
inline constexpr size_t ICE_PWD_MIN_LENGTH = 22;
inline constexpr size_t ICE_UFRAG_MAX_LENGTH = 256;
inline constexpr size_t ICE_PWD_MAX_LENGTH = 256;

// This is media-specific, so might belong
// somewhere like media/base/mediaconstants.h
inline constexpr int ICE_CANDIDATE_COMPONENT_RTP = 1;
inline constexpr int ICE_CANDIDATE_COMPONENT_RTCP = 2;
inline constexpr int ICE_CANDIDATE_COMPONENT_DEFAULT = 1;

// From RFC 4145, SDP setup attribute values.
inline constexpr char CONNECTIONROLE_ACTIVE_STR[] = "active";
inline constexpr char CONNECTIONROLE_PASSIVE_STR[] = "passive";
inline constexpr char CONNECTIONROLE_ACTPASS_STR[] = "actpass";
inline constexpr char CONNECTIONROLE_HOLDCONN_STR[] = "holdconn";

inline constexpr char LOCAL_TLD[] = ".local";

// Constants for time intervals are in milliseconds unless otherwise stated.
//
// Most of the following constants are the default values of IceConfig
// paramters. See IceConfig for detailed definition.
//
// Default value of IceConfig.receiving_timeout.
inline constexpr int RECEIVING_TIMEOUT = 2500;
// Default value IceConfig.ice_check_min_interval.
inline constexpr int MIN_CHECK_RECEIVING_INTERVAL = 50;
// The next two ping intervals are at the ICE transport level.
//
// STRONG_PING_INTERVAL is applied when the selected connection is both
// writable and receiving.
//
// Default value of IceConfig.ice_check_interval_strong_connectivity.
// (1000 * STUN_PING_PACKET_SIZE / 1000)
inline constexpr int STRONG_PING_INTERVAL = 480;
// WEAK_PING_INTERVAL is applied when the selected connection is either
// not writable or not receiving.
//
// Defaul value of IceConfig.ice_check_interval_weak_connectivity.
// (1000 * STUN_PING_PACKET_SIZE / 10000)
inline constexpr int WEAK_PING_INTERVAL = 48;
// The next two ping intervals are at the candidate pair level.
//
// Writable candidate pairs are pinged at a slower rate once they are stabilized
// and the channel is strongly connected.
inline constexpr int STRONG_AND_STABLE_WRITABLE_CONNECTION_PING_INTERVAL = 2500;
// Writable candidate pairs are pinged at a faster rate while the connections
// are stabilizing or the channel is weak.
inline constexpr int WEAK_OR_STABILIZING_WRITABLE_CONNECTION_PING_INTERVAL = 900;
// Default value of IceConfig.backup_connection_ping_interval
inline constexpr int BACKUP_CONNECTION_PING_INTERVAL = 25 * 1000;
// Default value of IceConfig.receiving_switching_delay.
inline constexpr int RECEIVING_SWITCHING_DELAY = 1000;
// Default value of IceConfig.regather_on_failed_networks_interval.
inline constexpr int REGATHER_ON_FAILED_NETWORKS_INTERVAL = 5 * 60 * 1000;
// Default value of IceConfig.ice_unwritable_timeout.
inline constexpr int CONNECTION_WRITE_CONNECT_TIMEOUT = 5 * 1000;  // 5 seconds
// Default value of IceConfig.ice_unwritable_min_checks.
inline constexpr uint32_t CONNECTION_WRITE_CONNECT_FAILURES = 5;  // 5 pings
// Default value of IceConfig.ice_inactive_timeout;
inline constexpr int CONNECTION_WRITE_TIMEOUT = 15 * 1000;  // 15 seconds
// Default vaule of IceConfig.stun_keepalive_interval;
inline constexpr int STUN_KEEPALIVE_INTERVAL = 10 * 1000;  // 10 seconds
// Default value of IceConfig.min_connection_lifetime;
inline constexpr int MIN_CONNECTION_LIFETIME = 10 * 1000;  // 10 seconds.
// A connection will be declared dead if it has not received anything for this
// long.
inline constexpr int DEAD_CONNECTION_RECEIVE_TIMEOUT = 30 * 1000;  // 30 seconds
// The timeout duration when a connection does not receive anything.
inline constexpr int WEAK_CONNECTION_RECEIVE_TIMEOUT = 2500;  // 2.5 seconds
// This is the length of time that we wait for a ping response to come back.
inline constexpr int CONNECTION_RESPONSE_TIMEOUT = 60 * 1000;  // 60 seconds

inline constexpr int MIN_PINGS_AT_WEAK_PING_INTERVAL = 3;

// The following constants are used at the candidate pair level to determine the
// state of a candidate pair.
//
// The type preference MUST be an integer from 0 to 126 inclusive.
// https://datatracker.ietf.org/doc/html/rfc5245#section-4.1.2.1
enum IcePriorityValue : uint8_t {
  ICE_TYPE_PREFERENCE_RELAY_TLS = 0,
  ICE_TYPE_PREFERENCE_RELAY_TCP = 1,
  ICE_TYPE_PREFERENCE_RELAY_UDP = 2,
  ICE_TYPE_PREFERENCE_PRFLX_TCP = 80,
  ICE_TYPE_PREFERENCE_HOST_TCP = 90,
  ICE_TYPE_PREFERENCE_SRFLX = 100,
  ICE_TYPE_PREFERENCE_PRFLX = 110,
  ICE_TYPE_PREFERENCE_HOST = 126
};

inline constexpr int kMaxTurnUsernameLength = 509;  // RFC 8489 section 14.3

}  // namespace webrtc

// Re-export symbols from the webrtc namespace for backwards compatibility.
// TODO(bugs.webrtc.org/4222596): Remove once all references are updated.
#ifdef WEBRTC_ALLOW_DEPRECATED_NAMESPACES
namespace cricket {
using ::webrtc::BACKUP_CONNECTION_PING_INTERVAL;
using ::webrtc::CN_AUDIO;
using ::webrtc::CN_DATA;
using ::webrtc::CN_OTHER;
using ::webrtc::CN_VIDEO;
using ::webrtc::CONNECTION_RESPONSE_TIMEOUT;
using ::webrtc::CONNECTION_WRITE_CONNECT_FAILURES;
using ::webrtc::CONNECTION_WRITE_CONNECT_TIMEOUT;
using ::webrtc::CONNECTION_WRITE_TIMEOUT;
using ::webrtc::CONNECTIONROLE_ACTIVE_STR;
using ::webrtc::CONNECTIONROLE_ACTPASS_STR;
using ::webrtc::CONNECTIONROLE_HOLDCONN_STR;
using ::webrtc::CONNECTIONROLE_PASSIVE_STR;
using ::webrtc::DEAD_CONNECTION_RECEIVE_TIMEOUT;
using ::webrtc::GROUP_TYPE_BUNDLE;
using ::webrtc::ICE_CANDIDATE_COMPONENT_DEFAULT;
using ::webrtc::ICE_CANDIDATE_COMPONENT_RTCP;
using ::webrtc::ICE_CANDIDATE_COMPONENT_RTP;
using ::webrtc::ICE_PWD_LENGTH;
using ::webrtc::ICE_PWD_MAX_LENGTH;
using ::webrtc::ICE_PWD_MIN_LENGTH;
using ::webrtc::ICE_TYPE_PREFERENCE_HOST;
using ::webrtc::ICE_TYPE_PREFERENCE_HOST_TCP;
using ::webrtc::ICE_TYPE_PREFERENCE_PRFLX;
using ::webrtc::ICE_TYPE_PREFERENCE_PRFLX_TCP;
using ::webrtc::ICE_TYPE_PREFERENCE_RELAY_TCP;
using ::webrtc::ICE_TYPE_PREFERENCE_RELAY_TLS;
using ::webrtc::ICE_TYPE_PREFERENCE_RELAY_UDP;
using ::webrtc::ICE_TYPE_PREFERENCE_SRFLX;
using ::webrtc::ICE_UFRAG_LENGTH;
using ::webrtc::ICE_UFRAG_MAX_LENGTH;
using ::webrtc::ICE_UFRAG_MIN_LENGTH;
using ::webrtc::IcePriorityValue;
using ::webrtc::LOCAL_TLD;
using ::webrtc::MIN_CHECK_RECEIVING_INTERVAL;
using ::webrtc::MIN_CONNECTION_LIFETIME;
using ::webrtc::MIN_PINGS_AT_WEAK_PING_INTERVAL;
using ::webrtc::RECEIVING_SWITCHING_DELAY;
using ::webrtc::RECEIVING_TIMEOUT;
using ::webrtc::REGATHER_ON_FAILED_NETWORKS_INTERVAL;
using ::webrtc::STRONG_AND_STABLE_WRITABLE_CONNECTION_PING_INTERVAL;
using ::webrtc::STRONG_PING_INTERVAL;
using ::webrtc::STUN_KEEPALIVE_INTERVAL;
using ::webrtc::WEAK_CONNECTION_RECEIVE_TIMEOUT;
using ::webrtc::WEAK_OR_STABILIZING_WRITABLE_CONNECTION_PING_INTERVAL;
using ::webrtc::WEAK_PING_INTERVAL;
}  // namespace cricket
#endif  // WEBRTC_ALLOW_DEPRECATED_NAMESPACES

#endif  // P2P_BASE_P2P_CONSTANTS_H_
