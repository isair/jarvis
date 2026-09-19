/*--
    health.h - broker telemetry declarations.
--*/

#pragma once

#include "broker.h"

VOID HealthMarkStart(BrokerContext* ctx);
VOID HealthWindowTick(BrokerContext* ctx);
VOID HealthHeartbeat(BrokerContext* ctx);
VOID HealthAdvance(BrokerContext* ctx, const TvmicPacketV1* hdr, int wrote);
