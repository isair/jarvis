/*--
    daemon_pipe.cpp - message-mode named pipe server for the Toustovač daemon.

    Protocol v1: the daemon sends one binary ``TvmicPacketV1`` per message
    (sample_count == 0 is treated as heartbeat). The broker replies with the
    same header shape carrying its counters. The PCM16 payload is the
    already-saturated int16 little-endian block produced by the daemon
    (48 kHz / mono / 16-bit = 96 kB/s + headers only).
--*/

#include "broker.h"

#include <stdio.h>
#include <string.h>

#define PIPE_SACL L"D:P(A;;GA;;;SY)(A;;GA;;;BA)(A;;GA;;;S-1-5-80)"

static const WCHAR kPipeName[] = L"\\\\.\\pipe\\ToustovacCleanMic.v1";

BOOL DaemonPipeCreate(BrokerContext* ctx)
{
    PSECURITY_DESCRIPTOR sd = NULL;
    SECURITY_ATTRIBUTES sa;

    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(
            PIPE_SACL, SDDL_REVISION_1, &sd, NULL)) {
        return FALSE;
    }
    sa.nLength = (DWORD)sizeof(sa);
    sa.lpSecurityDescriptor = sd;
    sa.bInheritHandle = FALSE;

    ctx->Pipe = CreateNamedPipeW(
        kPipeName,
        PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
        PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT |
            PIPE_REJECT_REMOTE_CLIENTS,
        1,                       /* single daemon connection */
        2048, 2048,
        500,                     /* read/write timeout 0.5 s */
        &sa);
    LocalFree(sd);
    return ctx->Pipe != NULL && ctx->Pipe != INVALID_HANDLE_VALUE;
}

BOOL DaemonPipePumpOnce(BrokerContext* ctx)
{
    char msg[BROKER_MAX_MESSAGE];
    ULONG got = 0;
    OVERLAPPED ov;

    memset(&ov, 0, sizeof(ov));
    ov.hEvent = CreateEventW(NULL, TRUE, FALSE, NULL);
    if (ov.hEvent == NULL) {
        return FALSE;
    }

    BOOL ok = ReadFile(ctx->Pipe, msg, sizeof(msg), &got, &ov);
    if (!ok && GetLastError() == ERROR_IO_PENDING) {
        if (!GetOverlappedResult(ctx->Pipe, &ov, &got, TRUE)) {
            CloseHandle(ov.hEvent);
            return FALSE;
        }
    }
    CloseHandle(ov.hEvent);
    if (got < TVMIC_PACKET_HEADER_SIZE) {
        return FALSE;
    }

    TvmicPacketV1 hdr;
    memcpy(&hdr, msg, sizeof(hdr));
    if (hdr.struct_size != TVMIC_PACKET_HEADER_SIZE ||
        hdr.protocol_version != TVMIC_PROTOCOL_VERSION ||
        hdr.sample_rate != TVMIC_SAMPLE_RATE ||
        hdr.channels != TVMIC_CHANNELS ||
        hdr.bits_per_sample != TVMIC_BITS_PER_SAMPLE) {
        BrokerEventLog(EVENTLOG_ERROR_TYPE, "protocol_mismatch");
        return FALSE;
    }

    /* PCM body (may be empty = heartbeat). */
    ULONG body = got - TVMIC_PACKET_HEADER_SIZE;
    const char* payload = msg + TVMIC_PACKET_HEADER_SIZE;

    if (hdr.sample_count == 0u) {
        /* heartbeat: >= 2 per second keeps the generation alive. */
        HealthHeartbeat(ctx);
        return TRUE;
    }
    if (body != hdr.sample_count * 2u) {
        BrokerEventLog(EVENTLOG_ERROR_TYPE, "size_mismatch body=%u want=%u",
                       body, hdr.sample_count * 2u);
        return FALSE;
    }

    if (!DriverWriteFrames(ctx, (const UCHAR*)payload, body)) {
        return FALSE;
    }
    ctx->Frames += hdr.sample_count / TVMIC_FRAME_SAMPLES;
    HealthAdvance(ctx, &hdr, 1);
    return TRUE;
}
