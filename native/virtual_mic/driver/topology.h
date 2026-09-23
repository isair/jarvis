/*--

Module Name:

    topology.h

Abstract:

    Topology static data for the Toustovač Clean Microphone: one microphone
    node and one WaveRT capture pin feeding into it. Modelled on the
    Microsoft SysVAD data-table pattern (see NOTICE). Modern PortCls types
    (PCNODE_DESCRIPTOR / PCCONNECTION_DESCRIPTOR / PCPIN_DESCRIPTOR).

--*/

#pragma once

#include <ntddk.h>
#include <portcls.h>
#include <ks.h>
#include <ksmedia.h>

// Node ids + counts (see topology.cpp).
#define TOPO_NODE_DEVICE_ID     0
#define TOPO_NODE_MIC_ID        1
#define TOPO_NUM_NODES          2
#define TOPO_NUM_CONNECTIONS    1
#define TOPO_NUM_PINS           1

// Count of data-range entries for the single 48 kHz / mono / PCM16 format.
#define NUM_SPEC_48_MONO        1

// ---------------------------------------------------------------------------
// Topology data defined in topology.cpp.
// ---------------------------------------------------------------------------

extern const KSDATARANGE_AUDIO  DataRange48Audio;

// ---------------------------------------------------------------------------
// Accessors called by the topology miniport.
// ---------------------------------------------------------------------------

PCNODE_DESCRIPTOR *GetTopologyNodes(_Out_ ULONG *pcbNodes);
PCCONNECTION_DESCRIPTOR *GetTopologyConnections(_Out_ ULONG *pcbConnections);
PCPIN_DESCRIPTOR *GetTopologyPins(_Out_ ULONG *pcbPins);
