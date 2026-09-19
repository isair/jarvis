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
// The single 48 kHz / mono / 16-bit format constraint.
// ---------------------------------------------------------------------------

static const LONG Spec48Mono[3] = { 1, 48000, 16 };  // mono, 48 kHz, 16 bits

const KSRESAMPLE_32_16_3 WaveFormat48kMono = { 2, 1, Spec48Mono };

// ---------------------------------------------------------------------------
// The single KS data range for the capture pin.
// ---------------------------------------------------------------------------

static const KSDATARANGE_AUDIO DataRange48Audio =
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

// ---------------------------------------------------------------------------
// Node / connection / pin tables (the one capture endpoint).
// ---------------------------------------------------------------------------

static NODE_DESCRIPTOR NodeTable[TOPO_NUM_NODES] = {
    { sizeof(NODE_DESCRIPTOR), TOPO_NODE_DEVICE_ID, KSNODETYPE_DEV_SPECIFIC, 1 },
    { sizeof(NODE_DESCRIPTOR), TOPO_NODE_MIC_ID, KSNODETYPE_MICROPHONE, 1 },
};

static CONNECTION_DESCRIPTOR ConnectionTable[TOPO_NUM_CONNECTIONS] = {
    // node 1 (mic) source pin 0 -> node 0 (device) destination pin 1.
    { sizeof(CONNECTION_DESCRIPTOR), 1, 0, 0, 1 },
};

static PIN_DESCRIPTOR PinTable[TOPO_NUM_PINS] = {
    {
        1,                                       /* pin id (eCapture)        */
        (WORD)(sizeof(WaveFormat48kMono) + 1) / sizeof(LONG),  /* fmt count */
        1,                                       /* one data format          */
        (ULONG_PTR)&DataRange48Audio,            /* data range start         */
        1,                                       /* data range count         */
        0,                                       /* connection matrix index  */
    },
};

// ---------------------------------------------------------------------------
// Property tables for the two-node, one-pin topology (int indices).
// ---------------------------------------------------------------------------

static const LONG NodeIds[TOPO_NUM_NODES] = { TOPO_NODE_DEVICE_ID,
                                              TOPO_NODE_MIC_ID };

static const PROPERTY_ITEM NodePropertyItems[TOPO_NUM_NODES] = {
    { 0, NodeIds },                       // node ids
    { 1, (LPCVOID)&WaveFormat48kMono },   // format id 0 -> the 48k format
};

static const PROPERTY_ITEM PinPropertyItems[TOPO_NUM_PINS] = {
    { 0, (LPCVOID)&DataRange48Audio },    // pin id 0 (pin 1) data ranges
};

// ---------------------------------------------------------------------------
// Accessors.
// ---------------------------------------------------------------------------

NODE_DESCRIPTOR *GetTopologyNodes(_Out_ ULONG *pcbNodes)
{
    *pcbNodes = TOPO_NUM_NODES;
    return NodeTable;
}

CONNECTION_DESCRIPTOR *GetTopologyConnections(_Out_ ULONG *pcbConnections)
{
    *pcbConnections = TOPO_NUM_CONNECTIONS;
    return ConnectionTable;
}

PIN_DESCRIPTOR *GetTopologyPins(_Out_ ULONG *pcbPins)
{
    *pcbPins = TOPO_NUM_PINS;
    return PinTable;
}
