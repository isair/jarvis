/*--

Module Name:

    broker.h

Abstract:

    Shared declarations of the broker service modules.

--*/

#pragma once

#include <windows.h>
#include <winsvc.h>
#include <sddl.h>
#include <setupapi.h>
#include "../driver/public/toustovac_virtual_mic_ioctl.h"

#define BROKER_NAME        L"ToustovacAudioBroker"

/* State names exactly as the spec dictates. */
#define BROKER_STATE_STARTING           "starting"
#define BROKER_STATE_DRIVER_READY       "driver_ready"
#define BROKER_STATE_PRODUCER_CONNECTED "producer_connected"
#define BROKER_STATE_STREAMING          "streaming"

/* Maximum accepted message size on the pipe (bounded): one frame at most. */
#define BROKER_MAX_MESSAGE (TVMIC_PACKET_HEADER_SIZE + (TVMIC_FRAME_SAMPLES * 2u) + 64u)

typedef struct _BrokerContext
{
    SERVICE_STATUS        ServiceStatus;
    SERVICE_STATUS_HANDLE StatusHandle;
    HANDLE                Pipe;         /* message-mode named pipe */
    HANDLE                Driver;       /* driver control handle   */
    const char*           State;
    ULONGLONG             StartedQpc;
    ULONG                 Frames;
    ULONG                 Silences;
    ULONG                 Drops;
    ULONG                 HeartbeatsIn1sWindow;
} BrokerContext;

VOID WINAPI BrokerServiceMain(DWORD argc, LPWSTR* argv);
VOID WINAPI BrokerCtrlHandler(DWORD ctrl);

/* driver_client.cpp */
BOOL DriverClientOpen(BrokerContext* ctx);
VOID DriverClientClose(BrokerContext* ctx);
BOOL DriverNegotiate(BrokerContext* ctx, const TvmicInit* init, TvmicInit* out);
BOOL DriverBeginGeneration(BrokerContext* ctx, ULONGLONG generation);
BOOL DriverEndGeneration(BrokerContext* ctx);
BOOL DriverWriteFrames(BrokerContext* ctx, const UCHAR* buf, ULONG len);
BOOL DriverSetMute(BrokerContext* ctx, LONG muted);
BOOL DriverQueryStatus(BrokerContext* ctx, TvmicCtlStatus* status);

/* daemon_pipe.cpp */
BOOL DaemonPipeCreate(BrokerContext* ctx);
BOOL DaemonPipePumpOnce(BrokerContext* ctx);

/* health.cpp */
VOID HealthMarkStart(BrokerContext* ctx);
VOID HealthWindowTick(BrokerContext* ctx);
VOID HealthHeartbeat(BrokerContext* ctx);
VOID HealthAdvance(BrokerContext* ctx, const TvmicPacketV1* hdr, int wrote);

/* eventlog.cpp */
VOID BrokerEventLog(WORD type, const char* fmt, ...);
