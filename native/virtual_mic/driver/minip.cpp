/*--
    minip.cpp - CPortInfo KS dispatch + CMiniportTopology bodies, plus the
    CMiniportWaveRT lifecycle. Modelled on Microsoft SysVAD (see NOTICE).
--*/

#include "minip.h"

// ---------------------------------------------------------------------------
// Int-valued property tables for the one-mic topology (0-based node ids).
// ---------------------------------------------------------------------------

static const LONG TopoNodeIds[TOPO_NUM_NODES] = { TOPO_NODE_DEVICE_ID,
                                                  TOPO_NODE_MIC_ID };
static const LONG TopoNodeCount[1]   = { TOPO_NUM_NODES };
static const LONG TopoPinCount[1]    = { TOPO_NUM_PINS };
static const LONG TopoConnCount[1]   = { TOPO_NUM_CONNECTIONS };
static const LONG TopoNameId[1]      = { TOPO_NODE_DEVICE_ID };

// Topology property set.
static const TVMIC_PROPERTY_ITEM TopoPropertyItems[] = {
    { KSPROPERTY_TOPOLOGY_NODES,        1, TopoNodeCount },
    { KSPROPERTY_TOPOLOGY_PINS,         1, TopoPinCount },
    { KSPROPERTY_TOPOLOGY_CONNECTIONS,  1, TopoConnCount },
    { KSPROPERTY_TOPOLOGY_NAME,         1, TopoNodeIds },
};

// Pin property tables, selected per pin id.
static const LONG PinIdItem[1]       = { TOPO_NODE_DEVICE_ID, TOPO_NODE_MIC_ID };
static const LONG PinCinstItem[1]    = { 1 };
static const TVMIC_PROPERTY_ITEM TopoNodeIdItem[1] = { { 0, 1, TopoNodeIds } };

// ---------------------------------------------------------------------------
// CPortInfo.
// ---------------------------------------------------------------------------

#pragma code_seg("PAGE")

NTSTATUS
CPortInfo::GetPropertyRange
(
    _In_  PNGUID  pPropertySet,
    _In_  ULONG   nPropId,
    _Out_ PLONG   pnMin,
    _Out_ PLONG   pnMax,
    _Out_ PMPI32  pStep
)
{
    if (pPropertySet == NULL || pnMin == NULL || pnMax == NULL ||
        pStep == NULL) {
        return STATUS_INVALID_PARAMETER;
    }

    *pnMin = 0;
    *pnMax = 0;
    pStep->Numerator = 1;
    pStep->Denominator = 0;

    if (IsEqualGUID(*pPropertySet, KSPROPERTYSETID_TOPOLOGY)) {
        switch (nPropId) {
        case KSPROPERTY_TOPOLOGY_NODES:
            *pnMin = 0;
            *pnMax = TOPO_NUM_NODES - 1;
            return STATUS_SUCCESS;
        case KSPROPERTY_TOPOLOGY_PINS:
            *pnMin = 1;
            *pnMax = TOPO_NUM_PINS;
            return STATUS_SUCCESS;
        default:
            break;
        }
        return STATUS_NOT_FOUND;
    }

    if (IsEqualGUID(*pPropertySet, KSPROPERTY_SET_ID_PIN? no)) {
    }

    return STATUS_NOT_FOUND;
}

NTSTATUS
CPortInfo::GetProperty
(
    _In_  PNGUID  pPropertySet,
    _In_  ULONG   nPropId,
    _In_  ULONG   nPropLen,
    _Out_writes_bytes_to_opt_(nPropLen, *PNPropLen) PVOID pProp,
    _Out_opt_ PULONG PNPropLen
)
{
    ULONG i;
    const TVMIC_PROPERTY_ITEM* items;
    ULONG nItems;

    if (pProp == NULL || nPropLen < sizeof(LONG)) {
        return STATUS_BUFFER_OVERFLOW;
    }

    if (IsEqualGUID(*pPropertySet, KSPROPERTYSETID_TOPOLOGY)) {
        items = TopoPropertyItems;
        nItems = (ULONG)(sizeof(TopoPropertyItems) /
                         sizeof(TopoPropertyItems[0]));
    } else {
        return STATUS_NOT_FOUND;
    }

    for (i = 0; i < nItems; i++) {
        if (items[i].nProperty == nPropId) {
            ULONG n = items[i].nItems;
            if (nPropLen < n * sizeof(LONG)) {
                return STATUS_BUFFER_OVERFLOW;
            }
            RtlCopyMemory(pProp, items[i].pData, n * sizeof(LONG));
            if (PNPropLen != NULL) {
                *PNPropLen = n * sizeof(LONG);
            }
            return STATUS_SUCCESS;
        }
    }
    return STATUS_NOT_FOUND;
}

#pragma code_seg()

// ---------------------------------------------------------------------------
// CMiniportTopology.
// ---------------------------------------------------------------------------

/* static */
NTSTATUS
CMiniportTopology::Create
(
    _In_ PNPAUDIO_DEVICE_CONTEXT DevCtx,
    _In_ ULONG MaxNameLen,
    _In_ PMINIPORT_TOPOLOGY_DESCRIPTOR Descriptor
)
{
    CMiniportTopology* pTopo;
    NTSTATUS status;

    if (Descriptor == NULL) {
        return STATUS_INVALID_PARAMETER;
    }

    pTopo = (CMiniportTopology*)ExAllocatePool2(PoolFlagPaged,
                                                sizeof(CMiniportTopology),
                                                'cimV');
    if (pTopo == NULL) {
        return STATUS_NO_MEMORY;
    }
    RtlZeroMemory(pTopo, sizeof(*pTopo));

    status = pTopo->Init(DevCtx, MaxNameLen, Descriptor);
    if (!NT_SUCCESS(status)) {
        ExFreePoolWithTag(pTopo, 'cimV');
        return status;
    }
    Descriptor[0].pTopoPort = (PMINIPORT)pTopo;
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

#pragma code_seg("PAGE")
NTSTATUS
CMiniportTopology::Init
(
    _In_ PNPAUDIO_DEVICE_CONTEXT DevCtx,
    _In_ ULONG MaxNameLen,
    _In_ PMINIPORT_TOPOLOGY_DESCRIPTOR Descriptor
)
{
    UNREFERENCED_PARAMETER(Descriptor);

    m_RefCount = 1;
    m_Device = DevCtx;
    m_nDevices = 1;
    m_MaxNameLen = MaxNameLen;
    m_nRanges = 0;
    return InitRanges();
}

NTSTATUS
CMiniportTopology::InitRanges()
{
    static CRANGES_INFORMATION sRanges = {0};
    m_pRangesInfo = &sRanges;
    return STATUS_SUCCESS;
}

VOID
CMiniportTopology::GetDescription(_Out_ PWSTR* ppwName)
{
    if (ppwName != NULL) {
        *ppwName = L"Toustovač Clean Microphone"; // static buffer, NUL-term.
    }
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
CMiniportTopology::GetProperty
(
    _In_  PNGUID  pPropertySet,
    _In_  ULONG   nPropId,
    _In_  ULONG   nPropLen,
    _Out_writes_bytes_to_opt_(nPropLen, *PNPropLen) PVOID pProp,
    _Out_opt_ PULONG PNPropLen
)
{
    return CPortInfo::GetProperty(pPropertySet, nPropId, nPropLen, pProp,
                                  PNPropLen);
}

NTSTATUS
CMiniportTopology::GetPropertyRange
(
    _In_  PNGUID pPropertySet,
    _In_  ULONG  nPropId,
    _Out_ PLONG  pnMin,
    _Out_ PLONG  pnMax,
    _Out_ PMPI32 pStep
)
{
    return CPortInfo::GetPropertyRange(pPropertySet, nPropId, pnMin, pnMax,
                                       pStep);
}

// ---------------------------------------------------------------------------
// CMiniportWaveRT lifecycle.
// ---------------------------------------------------------------------------

/* static */
NTSTATUS
CMiniportWaveRT::Create
(
    _In_ PNPAUDIO_DEVICE_CONTEXT DevCtx,
    _In_ ULONG MaxNameLen,
    _In_ PWCHAR PnpInterface
)
{
    CMiniportWaveRT* pWaveRt;
    NTSTATUS status;

    pWaveRt = (CMiniportWaveRT*)ExAllocatePool2(PoolFlagPaged,
                                                sizeof(CMiniportWaveRT),
                                                'cimV');
    if (pWaveRt == NULL) {
        return STATUS_NO_MEMORY;
    }
    RtlZeroMemory(pWaveRt, sizeof(*pWaveRt));

    status = pWaveRt->Init(DevCtx, MaxNameLen, PnpInterface);
    if (!NT_SUCCESS(status)) {
        ExFreePoolWithTag(pWaveRt, 'cimV');
        return status;
    }
    g_pWaveRt = pWaveRt;
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

#pragma code_seg("PAGE")
NTSTATUS
CMiniportWaveRT::Init
(
    _In_ PNPAUDIO_DEVICE_CONTEXT DevCtx,
    _In_ ULONG MaxNameLen,
    _In_ PWCHAR PnpInterface
)
{
    NTSTATUS status;

    m_RefCount = 1;
    m_nStreams = 0;
    m_nRegisteredProcesses = 0;
    m_nChannelCount = 1;
    m_nSamplesPerFrame = 480;
    m_MyBuffersAllocated = FALSE;
    m_nBufferSize = 0;
    m_pInterfaceId = (PWCHAR)PnpInterface;

    status = CPCMRing::Create(&g_pRing, TVMIC_RING_CAPACITY_FRAMES);
    if (!NT_SUCCESS(status)) {
        return status;
    }
    m_pRing = g_pRing;

    return InitRanges() ? STATUS_SUCCESS : m_Device != DevCtx
                       ? InitRanges(), (m_Device = DevCtx,
                       (m_nDevices = 1, STATUS_SUCCESS));
}

NTSTATUS
CMiniportWaveRT::InitRanges()
{
    static CRANGES_INFORMATION sRanges = {0};
    m_pRangesInfo = &sRanges;
    return STATUS_SUCCESS;
}

VOID
CMiniportWaveRT::GetDescription(_Out_ PWSTR* ppwName)
{
    if (ppwName != NULL) {
        *ppwName = L"Toustovač Clean Microphone"; // NUL-terminated
    }
}

VOID
CMiniportWaveRT::SetPowerState(_In_ POWER_STATE state)
{
    if (m_pRing != NULL && state != PowerDeviceD0) {
        m_pRing->Reset();
    }
}
#pragma code_seg()

// ---------------------------------------------------------------------------
// IMiniportWaveRT: one capture stream, PCM16 int ring in the DPC.
// ---------------------------------------------------------------------------

#include "minwavertstream.h"

#pragma code_seg("PAGE")
NTSTATUS
CMiniportWaveRT::CreateStream
(
    _In_ ULONG nStream,
    _In_ PVOID pPhysicalDevice,
    _In_ PMINIPORT_PROPERTY pProperty,
    _Out_ IMiniportWaveRTStream** PpStream
)
{
    NTSTATUS status;

    if (nStream >= MAX_NUMBER_OF_STREAMS || PpStream == NULL) {
        return STATUS_INVALID_PARAMETER;
    }

    status = CMiniportWaveRTStream::Create(
        this,
        pPhysicalDevice,
        pProperty,
        (PCMiniportWaveRTStream*)PpStream);
    return status;
}
#pragma code_seg()
