/*--
    minip.cpp - CPortInfo KS dispatch + CMiniportTopology bodies, plus the
    CMiniportWaveRT lifecycle. Modelled on Microsoft SysVAD (see NOTICE).
--*/

#include "minip.h"
#include "topology.h"

// ---------------------------------------------------------------------------
// Int-valued property tables for the one-mic topology (0-based node ids).
// ---------------------------------------------------------------------------

static const LONG TopoNodeIds[TOPO_NUM_NODES] = { TOPO_NODE_DEVICE_ID,
                                                  TOPO_NODE_MIC_ID };
static const LONG TopoNodeCount[1]   = { TOPO_NUM_NODES };
static const LONG TopoPinCount[1]    = { TOPO_NUM_PINS };
static const LONG TopoConnCount[1]   = { TOPO_NUM_CONNECTIONS };
static const LONG TopoNameId[1]      = { TOPO_NODE_DEVICE_ID };

// ---------------------------------------------------------------------------
// CPortInfo.
// ---------------------------------------------------------------------------

#pragma code_seg("PAGE")

NTSTATUS
CPortInfo::GetPropertyRange
(
    _In_ PPCPROPERTY_REQUEST PropertyRequest
)
{
    if (PropertyRequest == NULL) {
        return STATUS_INVALID_PARAMETER;
    }

    if (IsEqualGUID(*PropertyRequest->PropertyItem->Set,
                    KSPROPSETID_Topology)) {
        switch (PropertyRequest->PropertyItem->Id) {
        case KSPROPERTY_TOPOLOGY_NODES:
            if (PropertyRequest->ValueSize < sizeof(LONG))
                return STATUS_BUFFER_OVERFLOW;
            *(PLONG)PropertyRequest->Value = TOPO_NUM_NODES - 1;
            if (PropertyRequest->Irp)
                PropertyRequest->Irp->IoStatus.Information = sizeof(LONG);
            return STATUS_SUCCESS;
        case KSPROPERTY_TOPOLOGY_CATEGORIES:
            if (PropertyRequest->ValueSize < sizeof(LONG))
                return STATUS_BUFFER_OVERFLOW;
            *(PLONG)PropertyRequest->Value = TOPO_NUM_PINS;
            if (PropertyRequest->Irp)
                PropertyRequest->Irp->IoStatus.Information = sizeof(LONG);
            return STATUS_SUCCESS;
        default:
            break;
        }
        return STATUS_NOT_FOUND;
    }

    return STATUS_NOT_FOUND;
}

NTSTATUS
CPortInfo::GetProperty
(
    _In_ PPCPROPERTY_REQUEST PropertyRequest
)
{
    ULONG n;
    LPCVOID data;

    if (PropertyRequest == NULL || PropertyRequest->PropertyItem == NULL)
        return STATUS_INVALID_PARAMETER;

    if (IsEqualGUID(*PropertyRequest->PropertyItem->Set,
                    KSPROPSETID_Topology)) {
        switch (PropertyRequest->PropertyItem->Id) {
        case KSPROPERTY_TOPOLOGY_NODES:
            data = TopoNodeCount; n = 1; break;
        case KSPROPERTY_TOPOLOGY_CATEGORIES:
            data = TopoPinCount; n = 1; break;
        case KSPROPERTY_TOPOLOGY_CONNECTIONS:
            data = TopoConnCount; n = 1; break;
        case KSPROPERTY_TOPOLOGY_NAME:
            data = TopoNameId; n = 1; break;
        default:
            return STATUS_NOT_FOUND;
        }
    } else {
        return STATUS_NOT_FOUND;
    }

    if (PropertyRequest->Value == NULL) {
        if (PropertyRequest->Irp)
            PropertyRequest->Irp->IoStatus.Information = n * sizeof(LONG);
        return STATUS_SUCCESS;
    }
    if (PropertyRequest->ValueSize < n * sizeof(LONG))
        return STATUS_BUFFER_OVERFLOW;

    RtlCopyMemory(PropertyRequest->Value, data, n * sizeof(LONG));
    if (PropertyRequest->Irp)
        PropertyRequest->Irp->IoStatus.Information = n * sizeof(LONG);
    return STATUS_SUCCESS;
}

#pragma code_seg()

// ---------------------------------------------------------------------------
// Property handlers (PCPFNPROPERTY_HANDLER).
// ---------------------------------------------------------------------------

static NTSTATUS
HandlerPropTopology(_In_ PPCPROPERTY_REQUEST PropertyRequest)
{
    // The minor target is the CMiniportTopology instance.
    CMiniportTopology* pTopo = (CMiniportTopology*)PropertyRequest->MinorTarget;
    if (pTopo == NULL)
        return STATUS_INVALID_PARAMETER;
    return pTopo->GetProperty(PropertyRequest);
}

// Topology property set.
static const PCPROPERTY_ITEM TopoPropertyItems[] = {
    { &KSPROPSETID_Topology, KSPROPERTY_TOPOLOGY_NODES,
      PCPROPERTY_ITEM_FLAG_GET, HandlerPropTopology },
    { &KSPROPSETID_Topology, KSPROPERTY_TOPOLOGY_CATEGORIES,
      PCPROPERTY_ITEM_FLAG_GET, HandlerPropTopology },
    { &KSPROPSETID_Topology, KSPROPERTY_TOPOLOGY_CONNECTIONS,
      PCPROPERTY_ITEM_FLAG_GET, HandlerPropTopology },
    { &KSPROPSETID_Topology, KSPROPERTY_TOPOLOGY_NAME,
      PCPROPERTY_ITEM_FLAG_GET, HandlerPropTopology },
};

DEFINE_PCAUTOMATION_TABLE_PROP(AutomationTopology, TopoPropertyItems);

// ---------------------------------------------------------------------------
// CMiniportTopology.
// ---------------------------------------------------------------------------

/* static */
NTSTATUS
CMiniportTopology::Create
(
    _In_ PUNKNOWN UnknownAdapter,
    _In_ PRESOURCELIST ResourceList,
    _In_ PPORTTOPOLOGY Port,
    _Outptr_ PMINIPORTTOPOLOGY *Miniport
)
{
    CMiniportTopology* pTopo;
    NTSTATUS status;

    if (Port == NULL || Miniport == NULL) {
        return STATUS_INVALID_PARAMETER;
    }

    pTopo = (CMiniportTopology*)ExAllocatePool2(POOL_FLAG_NON_PAGED,
                                                sizeof(CMiniportTopology),
                                                'cimV');
    if (pTopo == NULL) {
        return STATUS_NO_MEMORY;
    }
    RtlZeroMemory(pTopo, sizeof(*pTopo));

    status = pTopo->Init(UnknownAdapter, ResourceList, Port);
    if (!NT_SUCCESS(status)) {
        ExFreePoolWithTag(pTopo, 'cimV');
        return status;
    }
    *Miniport = (PMINIPORTTOPOLOGY)pTopo;
    return STATUS_SUCCESS;
}

CMiniportTopology::~CMiniportTopology()
{
}

ULONG CMiniportTopology::AddRef()
{
    return (ULONG)InterlockedIncrement(&m_RefCount);
}

ULONG CMiniportTopology::Release()
{
    LONG ulRef = InterlockedDecrement(&m_RefCount);
    if (ulRef == 0) {
        delete this;
    }
    return (ULONG)ulRef;
}

NTSTATUS
CMiniportTopology::QueryInterface(_In_ REFGUID Guid, _Outptr_ PVOID *Object)
{
    if (Object == NULL)
        return STATUS_INVALID_PARAMETER;
    if (IsEqualGUID(Guid, IID_IMiniportTopology) ||
        IsEqualGUID(Guid, IID_IUnknown)) {
        *Object = (PVOID)this;
        AddRef();
        return STATUS_SUCCESS;
    }
    *Object = NULL;
    return STATUS_NOT_SUPPORTED;
}

#pragma code_seg("PAGE")
NTSTATUS
CMiniportTopology::Init
(
    _In_ PUNKNOWN UnknownAdapter,
    _In_ PRESOURCELIST ResourceList,
    _In_ PPORTTOPOLOGY Port
)
{
    UNREFERENCED_PARAMETER(UnknownAdapter);
    UNREFERENCED_PARAMETER(ResourceList);
    UNREFERENCED_PARAMETER(Port);

    m_RefCount = 1;
    m_Device = NULL;
    m_nDevices = 1;
    m_MaxNameLen = 0;
    m_nRanges = 0;
    return STATUS_SUCCESS;
}

NTSTATUS
CMiniportTopology::GetDescription(_Out_ PPCFILTER_DESCRIPTOR *ppwDescription)
{
    ULONG nNodes = 0, nConns = 0, nPins = 0;
    PCNODE_DESCRIPTOR *nodes = GetTopologyNodes(&nNodes);
    PCCONNECTION_DESCRIPTOR *conns = GetTopologyConnections(&nConns);
    PCPIN_DESCRIPTOR *pins = GetTopologyPins(&nPins);

    static PCFILTER_DESCRIPTOR Filter;
    Filter.Version = 0;
    Filter.AutomationTable = NULL;
    Filter.PinSize = sizeof(PCPIN_DESCRIPTOR);
    Filter.PinCount = nPins;
    Filter.Pins = pins;
    Filter.NodeSize = sizeof(PCNODE_DESCRIPTOR);
    Filter.NodeCount = nNodes;
    Filter.Nodes = nodes;
    Filter.ConnectionCount = nConns;
    Filter.Connections = conns;
    Filter.CategoryCount = 0;
    Filter.Categories = NULL;

    if (ppwDescription != NULL) {
        *ppwDescription = &Filter;
    }
    return STATUS_SUCCESS;
}

NTSTATUS
CMiniportTopology::DataRangeIntersection
(
    _In_ ULONG PinId,
    _In_ PKSDATARANGE DataRange,
    _In_ PKSDATARANGE MatchingDataRange,
    _In_ ULONG OutputBufferLength,
    _Out_writes_bytes_to_opt_(OutputBufferLength, *ResultantFormatLength) PVOID ResultantFormat,
    _Out_ PULONG ResultantFormatLength
)
{
    UNREFERENCED_PARAMETER(PinId);
    UNREFERENCED_PARAMETER(DataRange);
    UNREFERENCED_PARAMETER(MatchingDataRange);
    UNREFERENCED_PARAMETER(OutputBufferLength);
    UNREFERENCED_PARAMETER(ResultantFormat);
    UNREFERENCED_PARAMETER(ResultantFormatLength);
    return STATUS_NO_MATCH;
}

VOID
CMiniportTopology::SetPowerState(_In_ POWER_STATE state)
{
    UNREFERENCED_PARAMETER(state);
}

#pragma code_seg()

// ---------------------------------------------------------------------------
// IMiniportTopology implementation.
// ---------------------------------------------------------------------------

NTSTATUS
CMiniportTopology::GetProperty(_In_ PPCPROPERTY_REQUEST PropertyRequest)
{
    return CPortInfo::GetProperty(PropertyRequest);
}

NTSTATUS
CMiniportTopology::GetPropertyRange(_In_ PPCPROPERTY_REQUEST PropertyRequest)
{
    return CPortInfo::GetPropertyRange(PropertyRequest);
}

// ---------------------------------------------------------------------------
// CMiniportWaveRT lifecycle.
// ---------------------------------------------------------------------------

#include "minwavert.h"
#include "minwavertstream.h"

/* static */
NTSTATUS
CMiniportWaveRT::Create
(
    _In_ PUNKNOWN UnknownAdapter,
    _In_ PRESOURCELIST ResourceList,
    _In_ PPORTWAVERT Port,
    _Outptr_ PMINIPORTWAVERT *Miniport
)
{
    CMiniportWaveRT* pWaveRt;
    NTSTATUS status;

    if (Port == NULL || Miniport == NULL) {
        return STATUS_INVALID_PARAMETER;
    }

    pWaveRt = (CMiniportWaveRT*)ExAllocatePool2(POOL_FLAG_NON_PAGED,
                                                sizeof(CMiniportWaveRT),
                                                'cimV');
    if (pWaveRt == NULL) {
        return STATUS_NO_MEMORY;
    }
    RtlZeroMemory(pWaveRt, sizeof(*pWaveRt));

    status = pWaveRt->Init(UnknownAdapter, ResourceList, Port);
    if (!NT_SUCCESS(status)) {
        ExFreePoolWithTag(pWaveRt, 'cimV');
        return status;
    }
    g_pWaveRt = pWaveRt;
    *Miniport = (PMINIPORTWAVERT)pWaveRt;
    return status;
}

CMiniportWaveRT::~CMiniportWaveRT()
{
}

ULONG CMiniportWaveRT::AddRef()
{
    return (ULONG)InterlockedIncrement(&m_RefCount);
}

ULONG CMiniportWaveRT::Release()
{
    LONG ulRef = InterlockedDecrement(&m_RefCount);
    if (ulRef == 0) {
        delete this;
    }
    return (ULONG)ulRef;
}

NTSTATUS
CMiniportWaveRT::QueryInterface(_In_ REFGUID Guid, _Outptr_ PVOID *Object)
{
    if (Object == NULL)
        return STATUS_INVALID_PARAMETER;
    if (IsEqualGUID(Guid, IID_IMiniportWaveRT) ||
        IsEqualGUID(Guid, IID_IUnknown)) {
        *Object = (PVOID)this;
        AddRef();
        return STATUS_SUCCESS;
    }
    *Object = NULL;
    return STATUS_NOT_SUPPORTED;
}

#pragma code_seg("PAGE")
NTSTATUS
CMiniportWaveRT::Init
(
    _In_ PUNKNOWN UnknownAdapter,
    _In_ PRESOURCELIST ResourceList,
    _In_ PPORTWAVERT Port
)
{
    NTSTATUS status;

    UNREFERENCED_PARAMETER(UnknownAdapter);
    UNREFERENCED_PARAMETER(ResourceList);
    UNREFERENCED_PARAMETER(Port);

    m_RefCount = 1;
    m_nDevices = 1;
    m_MaxNameLen = 0;
    m_nRanges = 0;

    status = CPCMRing::Create(&g_pRing, TVMIC_RING_CAPACITY_FRAMES);
    if (!NT_SUCCESS(status)) {
        return status;
    }
    m_pRing = g_pRing;
    return STATUS_SUCCESS;
}

NTSTATUS
CMiniportWaveRT::GetDescription(_Out_ PPCFILTER_DESCRIPTOR *ppwDescription)
{
    ULONG nNodes = 0, nConns = 0, nPins = 0;
    PCNODE_DESCRIPTOR *nodes = GetTopologyNodes(&nNodes);
    PCCONNECTION_DESCRIPTOR *conns = GetTopologyConnections(&nConns);
    PCPIN_DESCRIPTOR *pins = GetTopologyPins(&nPins);

    static PCFILTER_DESCRIPTOR Filter;
    Filter.Version = 0;
    Filter.AutomationTable = NULL;
    Filter.PinSize = sizeof(PCPIN_DESCRIPTOR);
    Filter.PinCount = nPins;
    Filter.Pins = pins;
    Filter.NodeSize = sizeof(PCNODE_DESCRIPTOR);
    Filter.NodeCount = nNodes;
    Filter.Nodes = nodes;
    Filter.ConnectionCount = nConns;
    Filter.Connections = conns;
    Filter.CategoryCount = 0;
    Filter.Categories = NULL;

    if (ppwDescription != NULL) {
        *ppwDescription = &Filter;
    }
    return STATUS_SUCCESS;
}

NTSTATUS
CMiniportWaveRT::DataRangeIntersection
(
    _In_ ULONG PinId,
    _In_ PKSDATARANGE DataRange,
    _In_ PKSDATARANGE MatchingDataRange,
    _In_ ULONG OutputBufferLength,
    _Out_writes_bytes_to_opt_(OutputBufferLength, *ResultantFormatLength) PVOID ResultantFormat,
    _Out_ PULONG ResultantFormatLength
)
{
    UNREFERENCED_PARAMETER(PinId);
    UNREFERENCED_PARAMETER(DataRange);
    UNREFERENCED_PARAMETER(MatchingDataRange);
    UNREFERENCED_PARAMETER(OutputBufferLength);
    UNREFERENCED_PARAMETER(ResultantFormat);
    UNREFERENCED_PARAMETER(ResultantFormatLength);
    return STATUS_NO_MATCH;
}

NTSTATUS
CMiniportWaveRT::GetDeviceDescription(_Out_ PDEVICE_DESCRIPTION DeviceDescription)
{
    if (DeviceDescription != NULL) {
        RtlZeroMemory(DeviceDescription, sizeof(*DeviceDescription));
    }
    return STATUS_SUCCESS;
}

VOID
CMiniportWaveRT::SetPowerState(_In_ POWER_STATE state)
{
    if (m_pRing != NULL && state.DeviceState != PowerDeviceD0) {
        m_pRing->Reset();
    }
}
#pragma code_seg()

// ---------------------------------------------------------------------------
// IMiniportWaveRT: one capture stream, PCM16 int ring in the DPC.
// ---------------------------------------------------------------------------

#pragma code_seg("PAGE")
NTSTATUS
CMiniportWaveRT::NewStream
(
    _Out_ PMINIPORTWAVERTSTREAM *Stream,
    _In_ PPORTWAVERTSTREAM PortStream,
    _In_ ULONG Pin,
    _In_ BOOLEAN Capture,
    _In_ PKSDATAFORMAT DataFormat
)
{
    UNREFERENCED_PARAMETER(Pin);
    UNREFERENCED_PARAMETER(Capture);
    UNREFERENCED_PARAMETER(DataFormat);

    if (Stream == NULL) {
        return STATUS_INVALID_PARAMETER;
    }
    return CMiniportWaveRTStream::Create(this, m_pRing, PortStream,
                                         (PCMiniportWaveRTStream*)Stream);
}
#pragma code_seg()
