/*--

Module Name:

    topology.h

Abstract:

    Topology static data for the Toustovač Clean Microphone: one microphone
    node and one WaveRT capture pin feeding into it. Modelled on the
    Microsoft SysVAD data-table pattern (see NOTICE).

--*/

#pragma once

#include <windows.h>
#include <ks.h>
#include <ksmedia.h>
#include <portcls.h>

// Node ids + counts (see topology.cpp).
#define TOPO_NODE_DEVICE_ID     0
#define TOPO_NODE_MIC_ID        1
#define TOPO_NUM_NODES          2
#define TOPO_NUM_CONNECTIONS    1
#define TOPO_NUM_PINS           1

// Count of data-range entries for the single 48 kHz / mono / PCM16 format.
#define NUM_SPEC_48_MONO        1

// Property table entry shared by node/pin tables.
typedef struct _PROPERTY_ITEM {
    ULONG   nPropertyIndex;   // KSPROPERTY_*_ITEMS index value
    LPCVOID pData;            // pointer to the int/LONG item
} PROPERTY_ITEM, *PPROPERTY_ITEM;

// ---------------------------------------------------------------------------
// Topology data defined in topology.cpp.
// ---------------------------------------------------------------------------

// 48 kHz mono 16-bit PCM: {nCount, nItemSize, items[3]}.
typedef struct _KSRESAMPLE_32_16_3 {
    ULONG   nNum;     // 2  (nChannels + 1 = 1 channel entry + rate/bits)
    ULONG   nSize;    // 1  (LONG items in data-range)
    LPCVOID pItems;   // LONG[1] = { 48000 }
} KSRESAMPLE_32_16_3;

extern const KSRESAMPLE_32_16_3 WaveFormat48kMono;
extern const KSDATARANGE_AUDIO  DataRange48Audio;

// ---------------------------------------------------------------------------
// Accessors called by the topology miniport.
// ---------------------------------------------------------------------------

NODE_DESCRIPTOR *GetTopologyNodes(_Out_ ULONG *pcbNodes);
CONNECTION_DESCRIPTOR *GetTopologyConnections(_Out_ ULONG *pcbConnections);
PIN_DESCRIPTOR *GetTopologyPins(_Out_ ULONG *pcbPins);
