/*--
    Topology static data for the Toustovač Clean Microphone (capture-only).

      node 0  - the device node
      node 1  - the microphone node
      pin   1 - the WaveRT capture pin feeding node 1

    Exactly one canonical mix format: 48 kHz / mono / 16-bit PCM
    (480 frames / 10 ms), matching the CleanAudioBus format.
--*/

#include "topology.h"

// ---------------------------------------------------------------------------
// The single KS data range for the capture pin (48 kHz / mono / 16-bit PCM).
// ---------------------------------------------------------------------------

const KSDATARANGE_AUDIO DataRange48Audio =
{
    {
        (ULONG)sizeof(KSDATARANGE_AUDIO),
        0,
        0,
        0,
        STATIC_KSDATAFORMAT_TYPE_STREAM,
        KSDATAFORMAT_SUBTYPE_PCM,
        STATIC_KSDATAFORMAT_SPECIFIER_WAVEFORMATEX
    },
    1,          // maximum channels (mono)
    48000,      // minimum samples per second
    48000,      // maximum samples per second
    16,         // maximum bits per sample (int16)
};
static const PKSDATARANGE DataRangeList[]      = { (PKSDATARANGE)&DataRange48Audio };

// ---------------------------------------------------------------------------
// Node / connection / pin tables (the one capture endpoint), modern PC types.
// ---------------------------------------------------------------------------

static PCNODE_DESCRIPTOR NodeTable[TOPO_NUM_NODES] = {
    { 0, NULL, &KSNODETYPE_DEV_SPECIFIC, NULL },
    { 0, NULL, &KSNODETYPE_MICROPHONE,   NULL },
};

static PCCONNECTION_DESCRIPTOR ConnectionTable[TOPO_NUM_CONNECTIONS] = {
    // node 1 (mic) source pin 0 -> node 0 (device) destination pin 1.
    { 1, 0, 0, 1 },
};

static PCPIN_DESCRIPTOR PinTable[TOPO_NUM_PINS] = {
    {
        1, 1, 1, NULL,                 // MaxGlobal, MaxFilter, MinFilter, AutomationTable
        {
            0, NULL,                   // InterfacesCount, Interfaces
            0, NULL,                   // MediumsCount, Mediums
            1, DataRangeList,          // DataRangesCount, DataRanges
            KSPIN_DATAFLOW_IN,         // capture pin
            KSPIN_COMMUNICATION_NONE,
            &KSCATEGORY_AUDIO,
            NULL,                      // Name
            0                          // union: Reserved (LONGLONG)
        },
    },
};

// ---------------------------------------------------------------------------
// Accessors.
// ---------------------------------------------------------------------------

PCNODE_DESCRIPTOR *GetTopologyNodes(_Out_ ULONG *pcbNodes)
{
    *pcbNodes = TOPO_NUM_NODES;
    return NodeTable;
}

PCCONNECTION_DESCRIPTOR *GetTopologyConnections(_Out_ ULONG *pcbConnections)
{
    *pcbConnections = TOPO_NUM_CONNECTIONS;
    return ConnectionTable;
}

PCPIN_DESCRIPTOR *GetTopologyPins(_Out_ ULONG *pcbPins)
{
    *pcbPins = TOPO_NUM_PINS;
    return PinTable;
}
