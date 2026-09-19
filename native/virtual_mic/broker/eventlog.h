/*--
    eventlog.h - event log declarations.
--*/

#pragma once

#include "broker.h"

VOID BrokerEventLog(WORD type, const char* fmt, ...);
VOID WriteEventLogA_Line(HANDLE src, WORD type, const char* text);
