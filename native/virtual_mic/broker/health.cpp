/*--
    health.cpp - broker telemetry, heartbeat windows and silence fallback.
--*/

#include "broker.h"

#include <stdio.h>
#include <string.h>

static void WriteSilencePacket(BrokerContext* ctx)
{
    char buf[PACKET_HEADER_SIZE + TVMIC_FRAME_SAMPLES * 2u];
    TvmicPacketV1 hdr;

    memset(buf, 0, sizeof(buf));
    memset(&hdr, 0, sizeof(hdr));
    hdr.struct_size = PACKET_HEADER_SIZE;
    hdr.protocol_version = 1u;
    hdr.producer_generation = 0u;
    hdr.sequence = 0u;
    hdr.qpc_100ns = 0u;
    hdr.sample_rate = TVMIC_SAMPLE_RATE;
    hdr.channels = TVMIC_CHANNELS;
    hdr.bits_per_sample = TVMIC_BITS_PER_SAMPLE;
    hdr.sample_count = TVMIC_FRAME_SAMPLES;
    hdr.flags = TVMIC_FLAG_SILENCE;
    hdr.payload_crc32c = 0u;
    memcpy(buf, &hdr, sizeof(hdr));
    DriverWriteFrames(ctx, (const UCHAR*)buf, (ULONG)sizeof(buf));
}

VOID HealthMarkStart(BrokerContext* ctx)
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    ctx->StartedQpc = (ULONGLONG)li.QuadPart;
    if (ctx->State != NULL) {
        ctx->State = BROKER_STATE_DRIVER_READY;
    }
}

VOID HealthWindowTick(BrokerContext* ctx)
{
    /* one ~1 s window completed */
    if (ctx->State == NULL) {
        return;
    }
    if (ctx->HeartbeatsIn1sWindow >= 2u) {
        if (strcmp(ctx->State, BROKER_STATE_STARTING) == 0 ||
            strcmp(ctx->State, BROKER_STATE_DRIVER_READY) == 0) {
            ctx->State = BROKER_STATE_PRODUCER_CONNECTED;
        } else if (strcmp(ctx->State, BROKER_STATE_PRODUCER_CONNECTED) == 0) {
            ctx->State = BROKER_STATE_STREAMING;
        }
    } else if (strcmp(ctx->State, BROKER_STATE_STREAMING) == 0) {
        /* heartbeat expired: end the generation, flush and emit silence */
        DriverEndGeneration(ctx);
        WriteSilencePacket(ctx);
        ctx->State = BROKER_STATE_DRIVER_READY;
    }
    ctx->HeartbeatsIn1sWindow = 0u;
}

VOID HealthHeartbeat(BrokerContext* ctx)
{
    ctx->HeartbeatsIn1sWindow++;
    if (ctx->State != NULL &&
        strcmp(ctx->State, BROKER_STATE_STARTING) == 0) {
        ctx->State = BROKER_STATE_PRODUCER_CONNECTED;
    }
}

VOID HealthAdvance(BrokerContext* ctx, const TvmicPacketV1* hdr, int wrote)
{
    UNREFERENCED_PARAMETER(hdr);
    UNREFERENCED_PARAMETER(wrote);
    if (ctx->State != NULL && strcmp(ctx->State, BROKER_STATE_DRIVER_READY) == 0) {
        ctx->State = BROKER_STATE_PRODUCER_CONNECTED;
    }
}
