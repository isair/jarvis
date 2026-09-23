# Pinned toolchain (plan §12): Visual Studio 2026 / WDK 28000.2526 /
# Windows SDK 28000 / MSVC v14.50 / x64. The driver sources are WDK
# driver-package files built by msbuild (see driver/ToustovacVirtualMic.vcxproj);
# only the two user-mode binaries live in this CMake project.
#
# 1> Driver sources:   driver/        (msbuild ToustovacVirtualMic.vcxproj)
# 2> Broker sources:   broker/        (ToustovacAudioBroker.exe)
# 3> Installer source: installer/     (ToustovacAudioInstall.exe)

cmake_minimum_required(VERSION 3.23)
project(ToustovacAudioStack LANGUAGES CXX)

# /FS serializes parallel cl.exe PDB writes (C1041) under the MSBuild generator.
add_compile_options(/W4 /utf-8 /FS)

# --- broker service ---------------------------------------------------------
add_executable(ToustovacAudioBroker
    broker/service_main.cpp
    broker/broker_service.cpp
    broker/daemon_pipe.cpp
    broker/driver_client.cpp
    broker/health.cpp
    broker/eventlog.cpp
)
target_include_directories(ToustovacAudioBroker PRIVATE broker)

# --- install bootstrapper ---------------------------------------------------
add_executable(ToustovacAudioInstall installer/ToustovacAudioInstall.cpp)

foreach(t ToustovacAudioBroker ToustovacAudioInstall)
    set_target_properties(${t} PROPERTIES
        LINKER_LANGUAGE CXX
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach()

# import libraries for the SetupAPI / Config Manager entry points
target_link_libraries(ToustovacAudioBroker PRIVATE setupapi newdev cfgmgr32)
target_link_libraries(ToustovacAudioInstall PRIVATE setupapi newdev cfgmgr32)
