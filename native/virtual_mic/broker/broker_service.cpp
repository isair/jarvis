/*--
    broker_service.cpp - SCM entry, state machine and pipe server.

    State transitions (exactly per plan section 3):
      starting -> driver_ready -> producer_connected -> streaming
    Any producer loss / heartbeat expiry ends the generation, flushes the
    driver ring and falls back to silence.
--*/

#include "broker.h"
#include "eventlog.h"

#include <stdio.h>
#include <string.h>

static BrokerContext g_Ctx;

VOID WINAPI BrokerCtrlHandler(DWORD ctrl)
{
    if (g_Ctx.StatusHandle == NULL) {
        return;
    }
    switch (ctrl) {
    case SERVICE_CONTROL_STOP:
    case SERVICE_CONTROL_SHUTDOWN:
#if defined(SERVICE_CONTROL_PAGED_SYSTEM_SHUTDOWN)
    case SERVICE_CONTROL_PAGED_SYSTEM_SHUTDOWN:
#endif
        g_Ctx.ServiceStatus.dwCurrentState = SERVICE_STOP_PENDING;
        g_Ctx.ServiceStatus.dwWaitHint = 2000;
        SetServiceStatus(g_Ctx.StatusHandle, &g_Ctx.ServiceStatus);
        if (g_Ctx.Pipe != NULL) {
            DisconnectNamedPipe(g_Ctx.Pipe);
            CloseHandle(g_Ctx.Pipe);
            g_Ctx.Pipe = NULL;
        }
        g_Ctx.ServiceStatus.dwCurrentState = SERVICE_STOPPED;
        SetServiceStatus(g_Ctx.StatusHandle, &g_Ctx.ServiceStatus);
        break;
    case SERVICE_CONTROL_INTERROGATE:
        SetServiceStatus(g_Ctx.StatusHandle, &g_Ctx.ServiceStatus);
        break;
    default:
        break;
    }
}

VOID WINAPI BrokerServiceMain(DWORD argc, LPWSTR* argv)
{
    UNREFERENCED_PARAMETER(argc);
    UNREFERENCED_PARAMETER(argv);

    memset(&g_Ctx, 0, sizeof(g_Ctx));
    g_Ctx.State = BROKER_STATE_STARTING;

    g_Ctx.StatusHandle = RegisterServiceCtrlHandlerW(BROKER_NAME, BrokerCtrlHandler);
    if (g_Ctx.StatusHandle == NULL) {
        return;
    }
    g_Ctx.ServiceStatus.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
    g_Ctx.ServiceStatus.dwCurrentState = SERVICE_START_PENDING;
    g_Ctx.ServiceStatus.dwControlsAccepted =
        SERVICE_ACCEPT_STOP | SERVICE_ACCEPT_SHUTDOWN;
    SetServiceStatus(g_Ctx.StatusHandle, &g_Ctx.ServiceStatus);

    /* 1. open the sole driver-control handle. */
    if (!DriverClientOpen(&g_Ctx)) {
        g_Ctx.ServiceStatus.dwCurrentState = SERVICE_STOPPED;
        SetServiceStatus(g_Ctx.StatusHandle, &g_Ctx.ServiceStatus);
        return;
    }
    g_Ctx.State = BROKER_STATE_DRIVER_READY;
    BrokerEventLog(EVENTLOG_INFORMATION_TYPE,
                   "driver_ready protocol=%u", (unsigned)TVMIC_PROTOCOL_VERSION);

    /* 2. create the message-mode pipe with a security descriptor. */
    if (!DaemonPipeCreate(&g_Ctx)) {
        DriverClientClose(&g_Ctx);
        g_Ctx.ServiceStatus.dwCurrentState = SERVICE_STOPPED;
        SetServiceStatus(g_Ctx.StatusHandle, &g_Ctx.ServiceStatus);
        return;
    }
    g_Ctx.ServiceStatus.dwCurrentState = SERVICE_RUNNING;
    g_Ctx.ServiceStatus.dwWaitHint = 0;
    SetServiceStatus(g_Ctx.StatusHandle, &g_Ctx.ServiceStatus);
    HealthMarkStart(&g_Ctx);

    /* 3. pump loop: every message either flows PCM or falls back to silence.
     * Heartbeat expiry uses one 1 s window of at least two messages. */
    while (g_Ctx.ServiceStatus.dwCurrentState == SERVICE_RUNNING) {
        BOOL ok = DaemonPipePumpOnce(&g_Ctx);
        HealthWindowTick(&g_Ctx);
        UNREFERENCED_PARAMETER(ok);
    }

    DriverClientClose(&g_Ctx);
    if (g_Ctx.Pipe != NULL) {
        CloseHandle(g_Ctx.Pipe);
        g_Ctx.Pipe = NULL;
    }
    g_Ctx.ServiceStatus.dwCurrentState = SERVICE_STOPPED;
    SetServiceStatus(g_Ctx.StatusHandle, &g_Ctx.ServiceStatus);
}
