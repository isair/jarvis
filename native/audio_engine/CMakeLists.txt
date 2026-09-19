cmake_minimum_required(VERSION 3.24)
project(jarvis_audio_engine LANGUAGES C CXX)

add_compile_definitions(NOMINMAX)

# --------------------------------------------------------------------------
# Pinned source sets (no runtime downloads; configure-time fetch only when the
# vendored copies are absent). See NOTICE.
#
#   WebRTC:            https://webrtc.googlesource.com/src
#                      commit 4690e0b758acf196e0ed810b6cde77c3ca74e437
#                      (refs/heads/main@{#45047}, 2025-06-30)
#                      sparse dirs: api common_audio rtc_base system_wrappers
#                                   modules/audio_processing modules/third_party
#   chromium third_party:
#                      https://chromium.googlesource.com/chromium/src/third_party
#                      commit 05e78517da5dc3adea210622a08acf2cab7db595
#                      (sparse dirs: abseil-cpp, pffft)
#   perfetto:          https://chromium.googlesource.com/external/github.com/google/perfetto
#                      commit 5192b52b266bf5840929a30ac36037cf4f6eb2b7
#                      (perfetto_build_flags.h from build_configs/bazel)
#
# AEC3 processes 10 ms blocks; the engine enforces the per-frame order:
#   1) render (far-end) frame, 2) the matching capture frame, 3) cleaned out.
# --------------------------------------------------------------------------
set(WEBRTC_GIT        "https://webrtc.googlesource.com/src")
set(WEBRTC_COMMIT     "4690e0b758acf196e0ed810b6cde77c3ca74e437")
set(THIRDPARTY_GIT    "https://chromium.googlesource.com/chromium/src/third_party")
set(THIRDPARTY_COMMIT "05e78517da5dc3adea210622a08acf2cab7db595")
set(PERFETTO_GIT      "https://chromium.googlesource.com/external/github.com/google/perfetto.git")
set(PERFETTO_COMMIT   "5192b52b266bf5840929a30ac36037cf4f6eb2b7")

set(VENDOR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

if(EXISTS "${VENDOR}/webrtc-aec3/modules/audio_processing/aec3/echo_canceller3.h"
   AND EXISTS "${VENDOR}/third_party/abseil-cpp/absl/base/nullability.h"
   AND EXISTS "${VENDOR}/third_party/pffft/src/pffft.c"
   AND EXISTS "${VENDOR}/third_party/perfetto/include/perfetto/tracing/event_context.h"
   AND EXISTS "${VENDOR}/third_party/boringssl/src/include/openssl/ssl.h")
  set(WEBRTC_SRC "${VENDOR}/webrtc-aec3")
  set(ABS_L_ROOT "${VENDOR}/third_party/abseil-cpp")
  set(PFFFT_ROOT "${VENDOR}/third_party/pffft")
  set(SSL_INC    "${VENDOR}/third_party/boringssl/src/include")
  set(BSL_SRC    "${VENDOR}/third_party/boringssl/src")
  set(PERF_ROOT  "${VENDOR}/third_party/perfetto")
else()
  include(FetchContent)
  find_package(Git REQUIRED)
  FetchContent_Declare(webrtc GIT_REPOSITORY ${WEBRTC_GIT} GIT_TAG ${WEBRTC_COMMIT})
  FetchContent_Populate(webrtc)
  foreach(d api common_audio rtc_base system_wrappers modules/audio_processing modules/third_party
              modules/video_coding modules/audio_coding media/base common_video modules/rtp_rtcp modules/include)
    execute_process(COMMAND ${GIT_EXECUTABLE} -C "${webrtc_SOURCE_DIR}" sparse-checkout add ${d}
                    ERROR_QUIET OUTPUT_QUIET)
  endforeach()
  FetchContent_Declare(third_party GIT_REPOSITORY ${THIRDPARTY_GIT} GIT_TAG ${THIRDPARTY_COMMIT})
  FetchContent_Populate(third_party)
  foreach(d abseil-cpp pffft)
    execute_process(COMMAND ${GIT_EXECUTABLE} -C "${third_party_SOURCE_DIR}" sparse-checkout add ${d}
                    ERROR_QUIET OUTPUT_QUIET)
  endforeach()
  FetchContent_Declare(perfetto GIT_REPOSITORY ${PERFETTO_GIT} GIT_TAG ${PERFETTO_COMMIT})
  FetchContent_Populate(perfetto)
  set(WEBRTC_SRC "${webrtc_SOURCE_DIR}")
  set(ABS_L_ROOT "${third_party_SOURCE_DIR}/abseil-cpp")
  set(PFFFT_ROOT "${third_party_SOURCE_DIR}/pffft")
  set(SSL_INC    "${third_party_SOURCE_DIR}/boringssl/src/include")
  set(BSL_SRC    "${third_party_SOURCE_DIR}/boringssl/src")
  set(PERF_ROOT  "${perfetto_SOURCE_DIR}")
endif()

# --- generate the perfect proto code at configure time (from .proto) ---
if(NOT EXISTS "${PERF_ROOT}/protos/perfetto/trace/trace_packet.pbzero.h")
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(GEN_DIR "${CMAKE_BINARY_DIR}/perf_ds")
  file(MAKE_DIRECTORY "${GEN_DIR}")
  file(GLOB_RECURSE PERF_ALL "${PERF_ROOT}/protos/*.proto")
  # The merged bundles re-declare every individual file's symbols.
  set(PERF_PROTOS)
  foreach(p IN LISTS PERF_ALL)
    if(NOT p MATCHES "perfetto_config\\.proto$" AND
       NOT p MATCHES "perfetto_trace\\.proto$" AND
       NOT p MATCHES "perfetto_merged_metrics\\.proto$" AND
       NOT p MATCHES "all_chrome_metrics\\.proto$" AND
       NOT p MATCHES "all_webview_metrics\\.proto$")
      list(APPEND PERF_PROTOS "${p}")
    endif()
  endforeach()
  # Well-known google/protobuf protos shipped with grpc_tools.
  execute_process(COMMAND "${Python3_EXECUTABLE}" -c
    "import grpc_tools,os;print(os.path.join(os.path.dirname(grpc_tools.__file__),'_proto'))"
    OUTPUT_VARIABLE WKN_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
  list(APPEND PERF_PROTOS "${WKN_PATH}/google/protobuf/descriptor.proto")
  set(_ds_list)
  set(_i 0)
  foreach(p IN LISTS PERF_PROTOS)
    math(EXPR _i "${_i}+1")
    execute_process(COMMAND "${Python3_EXECUTABLE}" -m grpc_tools.protoc
                    "--proto_path=${PERF_ROOT}"
                    "--proto_path=${WKN_PATH}"
                    "--include_imports"
                    "--descriptor_set_out=${GEN_DIR}/${_i}.ds"
                    "${p}" RESULT_VARIABLE _pc)
    if(NOT _pc EQUAL 0)
      message(FATAL_ERROR "protoc descriptor set failed for ${p} (${_pc})")
    endif()
    list(APPEND _ds_list "${GEN_DIR}/${_i}.ds")
  endforeach()
  execute_process(COMMAND "${Python3_EXECUTABLE}"
                  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/gen_perf_protos.py"
                  ${_ds_list} "${PERF_ROOT}" RESULT_VARIABLE _gp)
  if(NOT _gp EQUAL 0)
    message(FATAL_ERROR "perf proto generation failed (${_gp})")
  endif()
endif()

set(MAP_DIR "${WEBRTC_SRC}/modules/audio_processing")

# --- closure of the APM pipeline, mirrored from upstream BUILD.gn ---
# WebRTC's BUILD.gn defines these targets; non-unittest sources only.
file(GLOB AEC3_SOURCES "${MAP_DIR}/aec3/*.cc" "${MAP_DIR}/utility/*.cc"
     "${MAP_DIR}/agc2/rnn_vad/*.cc")
list(FILTER AEC3_SOURCES EXCLUDE REGEX "_unittest\.cc$")
list(FILTER AEC3_SOURCES EXCLUDE REGEX "_neon\.cc$")
list(FILTER AEC3_SOURCES EXCLUDE REGEX "test_utils\.cc$")
list(FILTER AEC3_SOURCES EXCLUDE REGEX "rnn_vad_tool\.cc$")

file(GLOB AP_CORE "${MAP_DIR}/*.cc")
list(FILTER AP_CORE EXCLUDE REGEX "_unittest\.cc$")

file(GLOB AP_NS "${MAP_DIR}/ns/*.cc")
list(FILTER AP_NS EXCLUDE REGEX "_unittest\.cc$")
file(GLOB AP_AGC "${MAP_DIR}/agc/*.cc")
list(FILTER AP_AGC EXCLUDE REGEX "_unittest\.cc$")
file(GLOB AP_AGC2 "${MAP_DIR}/agc2/*.cc")
list(FILTER AP_AGC2 EXCLUDE REGEX "_unittest\.cc$")
list(FILTER AP_AGC2 EXCLUDE REGEX "/rnn_vad/.*")
file(GLOB AP_ED "${MAP_DIR}/echo_detector/*.cc" "${MAP_DIR}/ip/*.cc")
list(FILTER AP_ED EXCLUDE REGEX "_unittest\.cc$")
file(GLOB AP_AECM "${MAP_DIR}/aecm/*.cc" "${MAP_DIR}/aecm/*.c")
list(FILTER AP_AECM EXCLUDE REGEX "_mips\.cc$|_neon\.cc$")
list(FILTER AP_AECM EXCLUDE REGEX "unittest")
file(GLOB AP_AGCL "${MAP_DIR}/agc/legacy/*.cc")
list(FILTER AP_AGCL EXCLUDE REGEX "unittest")
file(GLOB AP_VAD "${MAP_DIR}/utility/*.cc")
list(FILTER AP_VAD EXCLUDE REGEX "_unittest\.cc$")
list(FILTER AP_VAD EXCLUDE REGEX "/n0$" )
list(FILTER AP_VAD EXCLUDE REGEX ".*")  # see AEC3_SOURCES (utility already there)
file(GLOB AP_CLL "${MAP_DIR}/capture_levels_adjuster/*.cc")
list(FILTER AP_CLL EXCLUDE REGEX "_unittest\.cc$")
file(GLOB AP_INC "${MAP_DIR}/include/*.cc")
list(FILTER AP_INC EXCLUDE REGEX "mock_.*\.cc$")
file(GLOB AP_LOG "${MAP_DIR}/logging/*.cc")
list(FILTER AP_LOG EXCLUDE REGEX "_unittest\.cc$")
file(GLOB AP_UTIL "${MAP_DIR}/utility/*.cc")
list(FILTER AP_UTIL EXCLUDE REGEX "_unittest\.cc$")
file(GLOB AP_DUMP "${MAP_DIR}/aec_dump/*.cc")
list(FILTER AP_DUMP EXCLUDE REGEX "_integration_test\.cc$")
list(FILTER AP_DUMP EXCLUDE REGEX "_unittest\.cc$")
# WEBRTC_APM_DEBUG_DUMP=0 builds use null_aec_dump_factory only.
list(FILTER AP_DUMP EXCLUDE REGEX "aec_dump_impl\.cc$")
list(FILTER AP_DUMP EXCLUDE REGEX "capture_stream_info\.cc$")
list(FILTER AP_DUMP EXCLUDE REGEX "mock_aec_dump\.cc$")

# common_audio C closure, minus the 11 isac codec files (not in this snapshot
# and not part of the APM pipeline; its .h are absent from the sparse tree).
file(GLOB CA_A "${WEBRTC_SRC}/common_audio/*.c"
              "${WEBRTC_SRC}/common_audio/third_party/*.c"
              "${WEBRTC_SRC}/common_audio/vad/*.c")
file(GLOB_RECURSE CA_C "${WEBRTC_SRC}/common_audio/signal_processing/*.c")
list(FILTER CA_C EXCLUDE REGEX "_mips\.c$")
list(FILTER CA_C EXCLUDE REGEX "_neon\.c$")
set(CA_CCC ${CA_A} ${CA_C})
file(GLOB_RECURSE CA_CC "${WEBRTC_SRC}/common_audio/*.cc")
list(FILTER CA_CC EXCLUDE REGEX "_unittest\.cc$")
list(FILTER CA_CC EXCLUDE REGEX "_benchmark\.cc$")
list(FILTER CA_CC EXCLUDE REGEX "_neon\.cc$")
list(FILTER CA_CC EXCLUDE REGEX "_mips\.cc$")
list(FILTER CA_CC EXCLUDE REGEX "/test/.*")
list(FILTER CA_CC EXCLUDE REGEX "/tests/.*")
list(FILTER CA_CC EXCLUDE REGEX "allocation_counter\.cc$")
list(APPEND CA_CCC ${CA_CC})

file(GLOB_RECURSE API_A "${WEBRTC_SRC}/api/*.cc")
list(FILTER API_A EXCLUDE REGEX "_unittest\.cc$")
list(FILTER API_A EXCLUDE REGEX "_benchmark\.cc$")
list(FILTER API_A EXCLUDE REGEX "/test/.*")
list(FILTER API_A EXCLUDE REGEX "/tests/.*")
# PeerConnection / NetEq / codec support files are not part of the APM
# closure and drag unrelated dependencies in.
list(FILTER API_A EXCLUDE REGEX "/api/.*peer_connection.*")
list(FILTER API_A EXCLUDE REGEX "/api/.*neteq.*")
list(FILTER API_A EXCLUDE REGEX "/api/audio_codecs/.*")
list(FILTER API_A EXCLUDE REGEX "/api/video_codecs/.*")
list(FILTER API_A EXCLUDE REGEX "/api/video/.*")
list(FILTER API_A EXCLUDE REGEX "/api/metronome/.*")
list(FILTER API_A EXCLUDE REGEX "/api/rtc_event_log/rtc_event\.cc$")
list(FILTER API_A EXCLUDE REGEX "/api/voip/voip_engine_factory\.cc$")
# Non-APM api files requiring the p2p / audio_device /fec stacks.
list(FILTER API_A EXCLUDE REGEX "/api/candidate\.cc$")
list(FILTER API_A EXCLUDE REGEX "/api/ice_transport_factory\.cc$")
list(FILTER API_A EXCLUDE REGEX "/api/rtp_parameters\.cc$")
list(FILTER API_A EXCLUDE REGEX "/api/create_peerconnection_factory\.cc$")
list(FILTER API_A EXCLUDE REGEX "/api/enable_media(_with_defaults)?\.cc$")
list(FILTER API_A EXCLUDE REGEX "/api/task_queue/task_queue_test\.cc$")
list(FILTER API_A EXCLUDE REGEX "/api/audio/create_audio_device_module\.cc$")
list(FILTER API_A EXCLUDE REGEX "/api/transport/.*")
list(FILTER API_A EXCLUDE REGEX "/api/rtc_event_log/.*")
list(FILTER API_A EXCLUDE REGEX "/default_task_queue_factory_(gcd|stdlib)\.cc$")
# only needed for tests; the factory cc drags pc/ into the closure
list(FILTER API_A EXCLUDE REGEX "frame_transformer_factory\.cc$")

file(GLOB_RECURSE RTCB_ALL "${WEBRTC_SRC}/rtc_base/*.cc")
# non-test leaf files that are part of the APM closure by dependency
file(GLOB API_V "${WEBRTC_SRC}/api/video/*.cc")
list(FILTER API_V EXCLUDE REGEX "_unittest.*\.cc$|_tests\.cc$")
# libyuv buffer TUs are required by frame_object.cc / transformer delegates.
# i010/i210/i410 remain excluded (no _16 variants provided; not referenced in
# the vendored APM closure). See minimal libyuv in third_party/libyuv.
list(FILTER API_V EXCLUDE REGEX "(i010|i210|i410)_buffer\.cc$")
list(FILTER API_V EXCLUDE REGEX "^$")
set(RTP_FT ${RTP_FT_LIST} ${API_V} "${WEBRTC_SRC}/api/rtp_packet_info.cc" "${WEBRTC_SRC}/api/units/time_delta.cc" "${WEBRTC_SRC}/api/units/timestamp.cc" "${WEBRTC_SRC}/modules/rtp_rtcp/source/rtp_sender_video_frame_transformer_delegate.cc" "${WEBRTC_SRC}/modules/rtp_rtcp/source/rtp_video_stream_receiver_frame_transformer_delegate.cc")
file(GLOB RTCPP "${WEBRTC_SRC}/modules/rtp_rtcp/source/rtcp_packet/*.cc")
list(FILTER RTCPP EXCLUDE REGEX "_unittest\.cc$")
list(APPEND RTP_FT ${RTCPP})
# remaining first-order deps of the video/rtp closure (all verified present)
list(APPEND RTP_FT
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/rtcp_packet.cc"
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/rtcp_packet/report_block.cc"
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/rtp_header_extensions.cc"
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/rtp_packet.cc"
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/rtp_packet_received.cc"
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/frame_object.cc"
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/rtp_generic_frame_descriptor.cc"
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/rtp_generic_frame_descriptor_extension.cc"
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/rtp_dependency_descriptor_extension.cc"
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/rtp_dependency_descriptor_reader.cc"
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/rtp_dependency_descriptor_writer.cc"
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/video_rtp_depacketizer.cc"
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/video_rtp_depacketizer_av1.cc"
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/video_rtp_depacketizer_generic.cc"
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/video_rtp_depacketizer_h264.cc"
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/video_rtp_depacketizer_h265.cc"
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/video_rtp_depacketizer_raw.cc"
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/video_rtp_depacketizer_vp8.cc"
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/video_rtp_depacketizer_vp9.cc"
  "${WEBRTC_SRC}/modules/video_coding/packet_buffer.cc"
  "${WEBRTC_SRC}/modules/video_coding/rtp_frame_reference_finder.cc"
  "${WEBRTC_SRC}/modules/video_coding/rtp_frame_id_only_ref_finder.cc"
  "${WEBRTC_SRC}/modules/video_coding/rtp_generic_ref_finder.cc"
  "${WEBRTC_SRC}/modules/video_coding/rtp_seq_num_only_ref_finder.cc"
  "${WEBRTC_SRC}/modules/video_coding/rtp_vp8_ref_finder.cc"
  "${WEBRTC_SRC}/modules/video_coding/rtp_vp9_ref_finder.cc"
  "${WEBRTC_SRC}/modules/video_coding/utility/decoded_frames_history.cc"
  "${WEBRTC_SRC}/api/transport/rtp/dependency_descriptor.cc"
  "${WEBRTC_SRC}/api/field_trials.cc"
  "${WEBRTC_SRC}/api/video_codecs/video_encoder.cc"
  "${WEBRTC_SRC}/api/video_codecs/video_decoder.cc")
list(APPEND RTP_FT
  "${WEBRTC_SRC}/rtc_base/ssl_adapter.cc"
  "${WEBRTC_SRC}/rtc_base/ssl_identity.cc"
  "${WEBRTC_SRC}/rtc_base/ssl_certificate.cc"
  "${WEBRTC_SRC}/rtc_base/ssl_stream_adapter.cc"
  "${WEBRTC_SRC}/rtc_base/ssl_fingerprint.cc"
  "${WEBRTC_SRC}/rtc_base/openssl_adapter.cc"
  "${WEBRTC_SRC}/rtc_base/openssl_stream_adapter.cc"
  "${WEBRTC_SRC}/rtc_base/openssl_digest.cc"
  "${WEBRTC_SRC}/rtc_base/openssl_utility.cc"
  "${WEBRTC_SRC}/rtc_base/openssl_session_cache.cc"
  "${WEBRTC_SRC}/rtc_base/boringssl_identity.cc"
  "${WEBRTC_SRC}/rtc_base/boringssl_certificate.cc"
  "${WEBRTC_SRC}/rtc_base/string_to_number.cc"
  "${WEBRTC_SRC}/rtc_base/string_utils.cc"
  "${WEBRTC_SRC}/api/audio/audio_processing.cc"
  "${WEBRTC_SRC}/api/video/video_frame.cc"
  "${WEBRTC_SRC}/api/video/video_frame_metadata.cc"
  "${WEBRTC_SRC}/api/video/rtp_video_frame_assembler.cc"
  "${WEBRTC_SRC}/api/video/frame_buffer.cc"
  "${WEBRTC_SRC}/api/rtp_parameters.cc"
  "${WEBRTC_SRC}/api/media_types.cc"
  "${WEBRTC_SRC}/api/transport/field_trial_based_config.cc"
  "${WEBRTC_SRC}/api/video_codecs/video_codec.cc"
  "${WEBRTC_SRC}/media/base/media_constants.cc"
  "${WEBRTC_SRC}/modules/video_coding/include/video_codec_interface.cc"
  "${WEBRTC_SRC}/modules/video_coding/utility/simulcast_rate_allocator.cc"
  "${WEBRTC_SRC}/modules/video_coding/svc/svc_rate_allocator.cc"
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/leb128.cc"
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/rtp_header_extension_map.cc"
  "${WEBRTC_SRC}/modules/rtp_rtcp/source/rtp_video_header.cc"
  "${WEBRTC_SRC}/common_video/h264/pps_parser.cc"
  "${WEBRTC_SRC}/common_video/h264/sps_parser.cc"
  "${WEBRTC_SRC}/common_video/h264/sps_vui_rewriter.cc"
   "${WEBRTC_SRC}/common_video/h265/h265_bitstream_parser.cc"
   "${WEBRTC_SRC}/common_video/h265/h265_sps_parser.cc"
   "${WEBRTC_SRC}/common_video/h265/h265_pps_parser.cc"
   "${WEBRTC_SRC}/common_video/h265/h265_vps_parser.cc"
   "${WEBRTC_SRC}/common_video/h265/h265_inline.cc"
   "${WEBRTC_SRC}/common_video/h264/h264_common.cc"
   "${WEBRTC_SRC}/common_video/h265/h265_common.cc"
   "${WEBRTC_SRC}/common_video/generic_frame_descriptor/generic_frame_info.cc"
   "${WEBRTC_SRC}/api/video_codecs/scalability_mode.cc"
   "${WEBRTC_SRC}/api/video_codecs/scalability_mode_helper.cc"
   "${WEBRTC_SRC}/api/video_codecs/simulcast_stream.cc"
   "${WEBRTC_SRC}/rtc_base/openssl_key_pair.cc"
   "${WEBRTC_SRC}/modules/audio_processing/agc/legacy/analog_agc.cc"
   "${WEBRTC_SRC}/modules/audio_processing/agc/legacy/digital_agc.cc")
# (RTCb re-adds happen after the RTCB_ALL filter chain below)
list(APPEND RTCB_ALL "${WEBRTC_SRC}/api/candidate.cc")
file(GLOB AP_VADD "${MAP_DIR}/vad/*.cc")
list(FILTER AP_VADD EXCLUDE REGEX "_unittest\.cc$")
file(GLOB CA_SQRT "${WEBRTC_SRC}/common_audio/third_party/spl_sqrt_floor/*.c")
list(FILTER CA_SQRT EXCLUDE REGEX "_mips\.c$|_neon\.c$|_arm\.c$")
file(GLOB API_RELC "${WEBRTC_SRC}/api/rtc_event_log/*.cc")
list(FILTER API_RELC EXCLUDE REGEX "_unittest\.cc$")
list(APPEND CA_CCC ${CA_SQRT})
list(APPEND RTP_FT ${AP_VADD} "${WEBRTC_SRC}/api/rtc_event_log/rtc_event_log.cc")
# Windows API libs for n0's n0 n0 n0 n0 n0 n0
list(APPEND API_RELC "${WEBRTC_SRC}/api/rtc_event_log/rtc_event_log_factory.cc")
list(APPEND WEBRTC_SOURCES ${API_RELC})
file(GLOB BSL_GEN_CC "${BSL_SRC}/gen/*.cc" "${BSL_SRC}/gen/*.h")
list(REMOVE_DUPLICATES RTCB_ALL)
list(FILTER RTCB_ALL EXCLUDE REGEX "_unittest\.cc$")
list(FILTER RTCB_ALL EXCLUDE REGEX "_benchmark\.cc$")
list(FILTER RTCB_ALL EXCLUDE REGEX "/test/.*")
list(FILTER RTCB_ALL EXCLUDE REGEX ".*_rust\.cc$")
list(FILTER RTCB_ALL EXCLUDE REGEX "/rtc_base/(mac_)?ifaddrs_converter\.cc$")
list(FILTER RTCB_ALL EXCLUDE REGEX "/rtc_base/(boringssl|.*boringssl).*\.cc$")
list(FILTER RTCB_ALL EXCLUDE REGEX "/rtc_base/crypto_random\.cc$")
list(FILTER RTCB_ALL EXCLUDE REGEX "/rtc_base/openssl.*\.cc$")
list(FILTER RTCB_ALL EXCLUDE REGEX "/rtc_base/ssl_.*\.cc$")
# test-only / platform-specific helpers not part of the APM closure
list(FILTER RTCB_ALL EXCLUDE REGEX "net_test_helpers\.cc$")
list(FILTER RTCB_ALL EXCLUDE REGEX "/rtc_base/strings/json\.cc$")
list(FILTER RTCB_ALL EXCLUDE REGEX "/rtc_base/system/warn_current_thread_is_deadlocked\.cc$")
# upstream picks exactly one task-queue backend; Windows gets the Win32 one
list(FILTER RTCB_ALL EXCLUDE REGEX "/rtc_base/task_queue_(gcd|stdlib)\.cc$")
# re-add the SSL/BoringSSL/crypto .cc explicitly after the filters
list(APPEND RTCB_ALL
  "${WEBRTC_SRC}/rtc_base/ssl_adapter.cc"
  "${WEBRTC_SRC}/rtc_base/ssl_identity.cc"
  "${WEBRTC_SRC}/rtc_base/ssl_certificate.cc"
  "${WEBRTC_SRC}/rtc_base/ssl_stream_adapter.cc"
  "${WEBRTC_SRC}/rtc_base/ssl_fingerprint.cc"
  "${WEBRTC_SRC}/rtc_base/openssl_adapter.cc"
  "${WEBRTC_SRC}/rtc_base/openssl_stream_adapter.cc"
  "${WEBRTC_SRC}/rtc_base/openssl_digest.cc"
  "${WEBRTC_SRC}/rtc_base/openssl_utility.cc"
  "${WEBRTC_SRC}/rtc_base/openssl_session_cache.cc"
  "${WEBRTC_SRC}/rtc_base/boringssl_identity.cc"
  "${WEBRTC_SRC}/rtc_base/boringssl_certificate.cc"
  "${WEBRTC_SRC}/rtc_base/string_to_number.cc"
  "${WEBRTC_SRC}/rtc_base/string_utils.cc"
  "${WEBRTC_SRC}/rtc_base/copy_on_write_buffer.cc"
  "${WEBRTC_SRC}/rtc_base/crypto_random.cc")
# legacy/non-win net stack pieces selected by webrtc.gni: for WEBRTC_WIN the
# physical-socket stack is used; the others are plain sources with #ifdefs.
file(GLOB_RECURSE SYSW_ALL "${WEBRTC_SRC}/system_wrappers/*.cc")
list(FILTER SYSW_ALL EXCLUDE REGEX "_unittest\.cc$")
list(FILTER SYSW_ALL EXCLUDE REGEX "/tests/.*")

# protobuf-generated .cc of the APM path
file(GLOB MTP_FFT "${WEBRTC_SRC}/modules/third_party/fft/*.c")
file(GLOB MTP_G711 "${WEBRTC_SRC}/modules/third_party/g711/*.c")

# jsoncpp 1.9.4 — upstream jsoncpp/jsoncpp @1.9.4, mirrored via raw fetch.
set(JSONCPP_INC "${VENDOR}/jsoncpp/include")
set(JSONCPP_SRC ${VENDOR}/jsoncpp/src/lib_json/json_reader.cpp
                ${VENDOR}/jsoncpp/src/lib_json/json_value.cpp
                ${VENDOR}/jsoncpp/src/lib_json/json_writer.cpp)

# isac_vad closure for modules/audio_processing/vad (upstream
# modules/audio_coding:isac_vad + isac_bwinfo):
set(ISAC_DIR "${WEBRTC_SRC}/modules/audio_coding/codecs/isac")
set(ISAC_SOURCES
  "${ISAC_DIR}/main/source/filter_functions.c"
  "${ISAC_DIR}/main/source/isac_vad.c"
  "${ISAC_DIR}/main/source/pitch_estimator.c"
  "${ISAC_DIR}/main/source/pitch_filter.c"
  "${ISAC_DIR}/bandwidth_info.h"
)
list(APPEND WEBRTC_SOURCES ${ISAC_SOURCES})

file(GLOB_RECURSE ABS_CC "${ABS_L_ROOT}/absl/*.cc" "${ABS_L_ROOT}/absl/*.c")
list(FILTER ABS_CC EXCLUDE REGEX "_test\.cc$")
list(FILTER ABS_CC EXCLUDE REGEX "_benchmark\.cc$")
list(FILTER ABS_CC EXCLUDE REGEX "test_common\.cc$")
list(FILTER ABS_CC EXCLUDE REGEX "benchmarks?\.cc$")
list(FILTER ABS_CC EXCLUDE REGEX "(exception_safety_testing|test_actions|test_helpers|test_matchers|scoped_mock_log|status_matchers|declarative_lookup|abi|no_destructor)\.cc$")
list(FILTER ABS_CC EXCLUDE REGEX "/absl/.*\.pb\.cc$")

# --- perfetto sources ---
# Nested tracing dirs: tracing/, tracing/core/, tracing/internal/, tracing/ipc/,
# tracing/ipc/{consumer,producer,service}/, tracing/service/ — all contribute to
# the in-process tracing closure (TracingMuxerImpl uses ProducerIPCClient /
# ConsumerIPCClient, SharedMemoryArbiter, InProcessSharedMemory, and the tracing
# service itself).
file(GLOB PERF_TRACING
     "${PERF_ROOT}/src/tracing/*.cc"
     "${PERF_ROOT}/src/tracing/core/*.cc"
     "${PERF_ROOT}/src/tracing/internal/*.cc"
     "${PERF_ROOT}/src/tracing/ipc/*.cc"
     "${PERF_ROOT}/src/tracing/ipc/consumer/*.cc"
     "${PERF_ROOT}/src/tracing/ipc/producer/*.cc"
     "${PERF_ROOT}/src/tracing/ipc/service/*.cc"
     "${PERF_ROOT}/src/tracing/service/*.cc")
file(GLOB PERF_PROTOZERO "${PERF_ROOT}/src/protozero/*.cc" "${PERF_ROOT}/src/protozero/filtering/*.cc")
file(GLOB PERF_BASE "${PERF_ROOT}/src/base/*.cc")
file(GLOB PERF_IPC "${PERF_ROOT}/src/ipc/*.cc")
file(GLOB PERF_ANDROID "${PERF_ROOT}/src/android_stats/*.cc")
file(GLOB_RECURSE PERF_GEN "${PERF_ROOT}/protos/*.gen.cc" "${PERF_ROOT}/protos/*.ipc.cc")
# Duplicate class names collapse into kmem.proto (covers kfree/kmalloc) and
# ftrace.proto (covers print). Keep the superset .cc per upstream merge rules.
list(FILTER PERF_GEN EXCLUDE REGEX "/(kfree|kmalloc)\.gen\.cc$")
list(FILTER PERF_GEN EXCLUDE REGEX "/print\.gen\.cc$")
list(FILTER PERF_TRACING EXCLUDE REGEX "_unittest\.cc$")
list(FILTER PERF_TRACING EXCLUDE REGEX "_benchmark\.cc$")
list(FILTER PERF_TRACING EXCLUDE REGEX "_fuzzer\.cc$")
list(FILTER PERF_TRACING EXCLUDE REGEX "integrationtest\.cc$")
list(FILTER PERF_TRACING EXCLUDE REGEX "/test/.*")
list(FILTER PERF_TRACING EXCLUDE REGEX "system_tracing_backend_fake\.cc$")
list(FILTER PERF_TRACING EXCLUDE REGEX "platform_fake\.cc$")
# tracing_muxer_fake.cc provides TracingMuxerFake::instance referenced by
# tracing_muxer_impl.cc (guarded at the single call site by build flags).
# relay_ipc_service.cc is referenced by service_ipc_host_impl.cc.
# zlib_compressor.cc needs the third-party zlib headers; not vendored here.
list(FILTER PERF_TRACING EXCLUDE REGEX "zlib_compressor\.cc$")
# Trace-writer test helper; not referenced by any runtime TU.
list(FILTER PERF_TRACING EXCLUDE REGEX "trace_writer_for_testing\.cc$")
# POSIX shmem backends (Windows uses shared_memory_windows.cc).
list(FILTER PERF_TRACING EXCLUDE REGEX "memfd\.cc$")
list(FILTER PERF_TRACING EXCLUDE REGEX "posix_shared_memory\.cc$")
# Relay client is only constructed by traced_relay (not part of the APM closure).
list(FILTER PERF_TRACING EXCLUDE REGEX "relay_ipc_client\.cc$")
# filter_util.cc requires google/protobuf/descriptor.h (full protobuf).
list(FILTER PERF_PROTOZERO EXCLUDE REGEX "filter_util\.cc$")
list(FILTER PERF_PROTOZERO EXCLUDE REGEX "_unittest\.cc$")
list(FILTER PERF_PROTOZERO EXCLUDE REGEX ".*_fuzzer\.cc$")
list(FILTER PERF_PROTOZERO EXCLUDE REGEX "_benchmark\.cc$")
list(FILTER PERF_PROTOZERO EXCLUDE REGEX "fake_scattered_buffer\.cc$")
list(FILTER PERF_PROTOZERO EXCLUDE REGEX "/test/.*")
list(FILTER PERF_BASE EXCLUDE REGEX "_unittest\.cc$")
list(FILTER PERF_BASE EXCLUDE REGEX "_benchmark\.cc$")
# POSIX-only file (cxxabi/dlfcn/pthread); upstream Windows builds skip it.
list(FILTER PERF_BASE EXCLUDE REGEX "debug_crash_stack_trace\.cc$")
list(FILTER PERF_BASE EXCLUDE REGEX "/test/.*")
list(FILTER PERF_IPC EXCLUDE REGEX "_unittest\.cc$")
list(FILTER PERF_IPC EXCLUDE REGEX "_benchmark\.cc$")
list(FILTER PERF_IPC EXCLUDE REGEX "integrationtest\.cc$")
list(FILTER PERF_IPC EXCLUDE REGEX ".*_fuzzer\.cc$")
list(FILTER PERF_IPC EXCLUDE REGEX "/test/.*")
list(FILTER PERF_IPC EXCLUDE REGEX "/protoc_plugin/.*")
# test-only support sources

# util/: standalone tools / gen helpers
list(FILTER PERF_BASE EXCLUDE REGEX "/util/.*\.cc$")
list(FILTER ABS_CC EXCLUDE REGEX "c_header_test\.c$")
list(FILTER ABS_CC EXCLUDE REGEX "print_hash_of\.cc$")
list(FILTER ABS_CC EXCLUDE REGEX "gaussian_distribution_gentables\.cc$")

set(PFFFT_C "${PFFFT_ROOT}/src/pffft.c")
set(LIBYUV_CCS "${WEBRTC_SRC}/third_party/libyuv/source/libyuv.cc")

set(JARVIS_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/engine.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/resampler.cpp"
)

set(WEBRTC_SOURCES
  ${JARVIS_SOURCES}
  ${AEC3_SOURCES}
  ${AP_CORE} ${AP_NS} ${AP_AGC} ${AP_AGC2} ${AP_ED} ${AP_VAD}
  ${AP_AECM} ${AP_AGCL}
  ${AP_CLL} ${AP_INC} ${AP_LOG} ${AP_UTIL} ${AP_DUMP}
  ${MTP_FFT} ${MTP_G711}
  "${VENDOR}/third_party/rnnoise/src/rnn_vad_weights.cc"
  ${CA_CCC} ${API_A} ${RTCB_ALL} ${SYSW_ALL} ${RTP_FT}
  "${PFFFT_C}" ${LIBYUV_CCS}
  ${PERF_TRACING} ${PERF_PROTOZERO} ${PERF_BASE} ${PERF_IPC} ${PERF_ANDROID}
  ${ABS_CC}
)
file(GLOB VC_SVC "${WEBRTC_SRC}/modules/video_coding/svc/*.cc")
list(FILTER VC_SVC EXCLUDE REGEX "_unittest\.cc$")
list(FILTER VC_SVC EXCLUDE REGEX "scalability_structure_test_helpers\.cc$")
list(APPEND WEBRTC_SOURCES ${ISAC_SOURCES} ${VC_SVC})
list(APPEND WEBRTC_SOURCES
  "${WEBRTC_SRC}/api/video_codecs/av1_profile.cc"
  "${WEBRTC_SRC}/api/video_codecs/h264_profile_level_id.cc"
  "${WEBRTC_SRC}/api/video_codecs/sdp_video_format.cc"
  "${WEBRTC_SRC}/api/video_codecs/scalability_mode.cc"
  "${WEBRTC_SRC}/api/video_codecs/vp9_profile.cc"
  "${WEBRTC_SRC}/modules/video_coding/utility/simulcast_utility.cc"
  "${WEBRTC_SRC}/video/config/video_encoder_config.cc")
list(REMOVE_DUPLICATES WEBRTC_SOURCES)
foreach(s IN LISTS WEBRTC_SOURCES)
  if(NOT EXISTS "${s}")
    message(FATAL_ERROR "Source missing at pinned commits: ${s}")
  endif()
endforeach()

add_library(jarvis_ae_obj OBJECT ${WEBRTC_SOURCES})
set_target_properties(jarvis_ae_obj PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON)
target_include_directories(jarvis_ae_obj PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${VENDOR}"
  "${WEBRTC_SRC}"
  "${ABS_L_ROOT}"
  "${PFFFT_ROOT}/src"
  "${PERF_ROOT}/include"
  "${PERF_ROOT}/include/perfetto"
  "${PERF_ROOT}/include/perfetto/base/build_configs/bazel"
  "${WEBRTC_SRC}/third_party/jsoncpp/includes"
  "${WEBRTC_SRC}/jsoncpp"
  "${WEBRTC_SRC}/third_party/libyuv/include"
  "${JSONCPP_INC}"
  "${SSL_INC}"
  "${BSL_SRC}"
  "${PERF_ROOT}"
)
# MSVC: the 256-bit types in <immintrin.h> carry operator[] only via the
# vector ABI enabled by /arch:AVX (and 256-bit ops by /arch:AVX2).
# _USE_MATH_DEFINES: M_PI for the vendored pffft.
target_compile_definitions(jarvis_ae_obj PUBLIC
  JARVIS_USE_WEBRTC_AEC3=1
  WEBRTC_APM_DEBUG_DUMP=0
  WEBRTC_WIN=1
  JSON_USE_EXCEPTION=0
  JSON_USE_NULLREF=0
  _USE_MATH_DEFINES)
# Winsock2-before-windows ordering for every TU (see src/pch.h).
target_precompile_headers(jarvis_ae_obj PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/pch.h")
# MSVC: 256-bit vector ABI for the *_avx2.cc kernels.
target_compile_options(jarvis_ae_obj PRIVATE /arch:AVX2 /utf-8 /permissive-)
# absl uses C-linkage names with namespaces (legal in its own builds); MSVC's
# /permissive- wants the tags repeated.
set_source_files_properties(${ABS_CC}
  PROPERTIES COMPILE_FLAGS "/Zc:__cplusplus-")
set_source_files_properties(
  "${MAP_DIR}/aec3/adaptive_fir_filter_avx2.cc"
  "${MAP_DIR}/aec3/adaptive_fir_filter_erl_avx2.cc"
  "${MAP_DIR}/aec3/fft_data_avx2.cc"
  "${MAP_DIR}/aec3/matched_filter_avx2.cc"
  "${MAP_DIR}/aec3/vector_math_avx2.cc"
  "${MAP_DIR}/agc2/rnn_vad/vector_math_avx2.cc"
  PROPERTIES COMPILE_FLAGS "/arch:AVX2")

# Generated perfetto proto TUs use the keyword-macro-free pch (src/pch_gen.h).
add_library(jarvis_perfgenc_obj OBJECT ${PERF_GEN})
set_target_properties(jarvis_perfgenc_obj PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON)
target_include_directories(jarvis_perfgenc_obj PUBLIC
  "${PERF_ROOT}/include"
  "${PERF_ROOT}/include/perfetto/base/build_configs/bazel"
  "${PERF_ROOT}"
)
target_compile_definitions(jarvis_perfgenc_obj PUBLIC JARVIS_USE_WEBRTC_AEC3=1 WEBRTC_WIN=1)
# ftrace_event.gen.cc has > 65k sections.
target_compile_options(jarvis_perfgenc_obj PRIVATE /bigobj)
target_precompile_headers(jarvis_perfgenc_obj PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/pch_gen.h")

add_library(jarvis_jsoncpp_obj OBJECT ${JSONCPP_SRC})
set_target_properties(jarvis_jsoncpp_obj PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON)
target_include_directories(jarvis_jsoncpp_obj PUBLIC "${JSONCPP_INC}")
target_compile_definitions(jarvis_jsoncpp_obj PUBLIC
  JSON_USE_EXCEPTION=0 JSON_USE_NULLREF=0)

# BoringSSL, pure-C mode; provides openssl/*.h + crypto/ssl objects.
file(GLOB_RECURSE BSL_CC "${BSL_SRC}/crypto/*.cc" "${BSL_SRC}/ssl/*.cc"
     "${BSL_SRC}/decrepit/*.cc" "${BSL_SRC}/gen/*.cc")
list(FILTER BSL_CC EXCLUDE REGEX "/test/.*")
list(FILTER BSL_CC EXCLUDE REGEX "_test\.cc$")
list(FILTER BSL_CC EXCLUDE REGEX "/fuzz/.*")
list(FILTER BSL_CC EXCLUDE REGEX "/util/.*")
add_library(jarvis_bsl_obj OBJECT ${BSL_CC})
set_target_properties(jarvis_bsl_obj PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON)
target_include_directories(jarvis_bsl_obj PUBLIC "${BSL_SRC}/include" "${BSL_SRC}")
target_compile_definitions(jarvis_bsl_obj PUBLIC OPENSSL_NO_ASM WIN32_LEAN_AND_MEAN)
# PCH (created by jarvis_ae_obj under /arch:AVX2) — keep option parity.
target_compile_options(jarvis_bsl_obj PRIVATE /arch:AVX2 /utf-8 /permissive-)
target_precompile_headers(jarvis_bsl_obj REUSE_FROM jarvis_ae_obj)

add_library(jarvis_audio_engine SHARED
  "${CMAKE_CURRENT_SOURCE_DIR}/jarvis_audio_engine.def"
  $<TARGET_OBJECTS:jarvis_ae_obj>
  $<TARGET_OBJECTS:jarvis_perfgenc_obj>
  $<TARGET_OBJECTS:jarvis_jsoncpp_obj>
  $<TARGET_OBJECTS:jarvis_bsl_obj>)
set_target_properties(jarvis_audio_engine PROPERTIES
  OUTPUT_NAME "jarvis_audio_engine" CXX_STANDARD 20)
target_link_libraries(jarvis_audio_engine PRIVATE ole32 avrt iphlpapi winmm)

add_executable(jarvis_audio_engine_sidecar
  "${CMAKE_CURRENT_SOURCE_DIR}/src/sidecar_main.cpp"
  $<TARGET_OBJECTS:jarvis_ae_obj>
  $<TARGET_OBJECTS:jarvis_perfgenc_obj>
  $<TARGET_OBJECTS:jarvis_jsoncpp_obj>
  $<TARGET_OBJECTS:jarvis_bsl_obj>)
set_target_properties(jarvis_audio_engine_sidecar PROPERTIES
  OUTPUT_NAME "jarvis_audio_engine_sidecar" CXX_STANDARD 20)
target_include_directories(jarvis_audio_engine_sidecar PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(jarvis_audio_engine_sidecar PRIVATE
  JARVIS_USE_WEBRTC_AEC3=1 WEBRTC_WIN=1)
target_link_libraries(jarvis_audio_engine_sidecar PRIVATE ole32 avrt iphlpapi winmm)
