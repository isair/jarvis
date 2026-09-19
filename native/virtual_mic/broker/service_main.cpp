/*--
    service_main.cpp - Toustovač audio broker service entry point.

    Owns (a) the sole writable driver-control handle, (b) the versioned
    named-pipe server for the Toustovač daemon, and (c) the silence-on-loss
    timers. No DSP, no source selection, no parsing of the Voice PE protocol.
--*/

#include <windows.h>
#include <stdio.h>

#include "broker.h"

int __cdecl main(int argc, char** argv)
{
    static SERVICE_TABLE_ENTRYW table[] = {
        { (LPWSTR) L"ToustovacAudioBroker", BrokerServiceMain },
        { NULL, NULL }
    };

    UNREFERENCED_PARAMETER(argc);
    UNREFERENCED_PARAMETER(argv);

    StartServiceCtrlDispatcherW(table);
    return 0;
}
