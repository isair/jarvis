/*--
    eventlog.cpp - Windows Event Log for the broker (no PCM contents).
--*/

#include "broker.h"
#include "eventlog.h"

#include <stdio.h>
#include <stdarg.h>
#include <string.h>

static HANDLE g_EventLog = NULL;

VOID BrokerEventLog(WORD type, const char* fmt, ...)
{
    char line[512];
    va_list args;

    if (g_EventLog == NULL) {
        g_EventLog = RegisterEventSourceW(BROKER_NAME, NULL);
    }
    if (g_EventLog == NULL) {
        return;
    }

    va_start(args, fmt);
    _vsnprintf(line, sizeof(line), fmt, args);
    va_end(args);

    WriteEventLogA_Line(g_EventLog, type, line);
}

VOID WriteEventLogA_Line(HANDLE src, WORD type, const char* text)
{
    LPCWSTR strings[1];
    WCHAR wline[512];
    int i;

    for (i = 0; i < 511 && text != NULL && text[i]; ++i) {
        wline[i] = (WCHAR)(unsigned char)text[i];
    }
    wline[i] = L'\0';
    strings[0] = wline;
    /* hEventLog, type, category, id, sid, numStrings, dataSize, strings, raw */
    ReportEventW(src, type, 0, 0, NULL, 1, 0, strings, NULL);
}
