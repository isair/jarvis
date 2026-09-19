/*--
    daemon_pipe.h - named-pipe server declarations.
--*/

#pragma once

#include "broker.h"

BOOL DaemonPipeCreate(BrokerContext* ctx);
BOOL DaemonPipePumpOnce(BrokerContext* ctx);
