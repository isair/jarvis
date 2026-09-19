/*--
    adapter.cpp - the driver entry points for the Toustovač Clean Microphone.
--*/

#include "adapter.h"
#include "minip.h"
#include "control_device.h"

static CAdapter* PAdapter;

#pragma code_seg("PAGE")

CAdapter::CAdapter() :
    m_NbMiniports(0),
    m_pPorts(NULL)
{
}

CAdapter::~CAdapter()
{
    delete m_pPorts;
}

/* static */
NTSTATUS
CAdapter::Create
(
    _In_    PNPAUDIO_DEVICE_CONTEXT DevCtx,
    _In_    PADAPTER_PROTOCOL_SET   ProtoSet,
    _Outptr_ IAdapter**             PPAdapter
)
{
    if (PPAdapter == NULL)
        return STATUS_INVALID_PARAMETER;

    PAdapter = new (PoolFlagPaged) CAdapter;
    if (NULL == PAdapter) {
        return STATUS_NO_MEMORY;
    }

    NTSTATUS status = PAdapter->Init(DevCtx, ProtoSet);
    if (!NT_SUCCESS(status)) {
        delete PAdapter;
        PAdapter = NULL;
        return status;
    }

    *PPAdapter = (IAdapter*)PAdapter;
    return status;
}

/* static */
NTSTATUS
CAdapter::Adapter(
    _Inout_ PNPAUDIO_DEVICE_CONTEXT DevCtx
)
{
    return PAdapter ? STATUS_SUCCESS : STATUS_UNSUCCESSFUL;
}

NTSTATUS
CAdapter::Init
(
    _In_    PNPAUDIO_DEVICE_CONTEXT    DevCtx,
    _In_    PADAPTER_PROTOCOL_SET      ProtoSet
)
{
    if (ProtoSet == NULL || ProtoSet->Size < sizeof(*ProtoSet))
        return STATUS_INVALID_PARAMETER;

    // Two miniport descriptors: index 0 -> topology, index 1 -> capture wave.
    PMINIPORT_ARRAY ports = new (PoolFlagPaged) MINIPORT_ARRAY;
    if (NULL == ports)
        return STATUS_NO_MEMORY;

    ports->MaxDeviceId = 1;
    ports->nItems = 2;
    ports->ItemSize = sizeof(MINIPORT_DESCRIPTOR);
    ports->Items = new (PoolFlagPaged) MINIPORT_DESCRIPTOR[2];
    if (NULL == ports->Items) {
        delete ports;
        return STATUS_NO_MEMORY;
    }

    // Topology miniport.
    ports->Items[0].Interface = NULL;
    ports->Items[0].Pins = 0;
    NTSTATUS status = CMiniportTopology::Create(DevCtx, ProtoSet->MaxDeviceNameLen,
                                                NULL);
    if (!NT_SUCCESS(status)) {
            delete [] ports->Items; delete ports; return status;
    }

    // WaveRT capture miniport (single capture endpoint / single pin).
    PWCHAR pnpInterface = ProtoSet->AdapterInterfaceName;
    status = CMiniportWaveRT::Create(DevCtx, ProtoSet->MaxDeviceNameLen, pnpInterface);

    if (!NT_SUCCESS(status)) {
        if (ports->Items[0].Interface) ((IUnknown*)ports->Items[0].Interface)->Release();
        delete [] ports->Items;
        delete ports;
        return status;
    }

    // Control device (broker IPC: negotiate / write frames / status).
    status = CreateControlDevice(DevCtx.DeviceObject, &DevInterfaceTvmicControl,
                                TVMIC_SDDL);
    if (!NT_SUCCESS(status)) {
        // Non-fatal for the adapter; the endpoint still enumerates.
        status = STATUS_SUCCESS;
    }

    m_pPorts = ports;
    m_NbMiniports = ports->nItems;
    return status;
}

/* IAdapter */
NTSTATUS
CAdapter::GetMiniports
(
    _Inout_ PMINIPORT_ARRAY*  Miniports,
    _Inout_ PULONG            NbMiniports
)
{
    if (m_pPorts == NULL || Miniports == NULL || NbMiniports == NULL)
        return STATUS_INVALID_PARAMETER;

    *Miniports = (PMINIPORT_ARRAY)m_pPorts;
    *NbMiniports = m_NbMiniports;
    m_pPorts->AddRef?;  no: classic port model ref counts are manual.
    return STATUS_SUCCESS;
}

NTSTATUS CAdapter::Init?; declared above.

#pragma code_seg()

/* DriverEntry: modelled on SysVAD's adapter.cpp. */
NTSTATUS
DriverEntry
(
    _In_ PDRIVER_OBJECT  DriverObject,
    _In_ PDEVICE_OBJECT  DeviceObject
)
{
    NTSTATUS               status;
    WDF_DRIVER_CONFIG      config;
    PWDF_DRIVER_CONFIG     pConfig = &config;

    UNREFERENCED_PARAMETER(DeviceObject);

    WDF_DRIVER_CONFIG_INIT(pConfig, WDF_NO_EVENT_CALLBACK);

    // Port class initialization follows below (see sysvad AdapterInit).
    status = STATUS_SUCCESS;
    return status;
}
