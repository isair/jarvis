/*--
    adapter.cpp - the driver entry points for the Toustovač Clean Microphone.
    Modern PortCls model: DriverEntry -> PcInitializeAdapterDriver;
    AddDevice -> PcAddAdapterDevice; StartDevice creates the topology +
    WaveRT miniports, registers them as subdevices, and registers the
    private control device interface.
--*/

#include "adapter.h"
#include "minip.h"
#include "minwavert.h"
#include "control_device.h"

#define MIMETYPE_TOPOLOGY_ID   L"0"
#define MIMETYPE_WAVE_PORT_ID  L"1"

static CAdapter* PAdapter;

// ---------------------------------------------------------------------------
// CAdapter (IAdapterPnpManagement).
// ---------------------------------------------------------------------------

#pragma code_seg("PAGE")
CAdapter::CAdapter() :
    m_RefCount(1),
    m_Device(NULL)
{
}

CAdapter::~CAdapter()
{
}

/* static */
NTSTATUS
CAdapter::Create
(
    _Inout_ PDEVICE_OBJECT DeviceObject,
    _Outptr_ IAdapterPnpManagement **PPAdapter
)
{
    if (PPAdapter == NULL)
        return STATUS_INVALID_PARAMETER;

    PAdapter = new (NonPagedPoolNx) CAdapter;
    if (PAdapter == NULL)
        return STATUS_NO_MEMORY;

    NTSTATUS status = PAdapter->Init(DeviceObject);
    if (!NT_SUCCESS(status)) {
        delete PAdapter;
        PAdapter = NULL;
        return status;
    }
    *PPAdapter = (IAdapterPnpManagement*)PAdapter;
    return STATUS_SUCCESS;
}

NTSTATUS
CAdapter::Init(_Inout_ PDEVICE_OBJECT DeviceObject)
{
    m_Device = DeviceObject;
    return STATUS_SUCCESS;
}

ULONG CAdapter::AddRef()   { return (ULONG)InterlockedIncrement(&m_RefCount); }
ULONG CAdapter::Release()
{
    LONG n = InterlockedDecrement(&m_RefCount);
    if (n == 0) delete this;
    return (ULONG)n;
}

NTSTATUS
CAdapter::QueryInterface(_In_ REFGUID Guid, _Outptr_ PVOID *Object)
{
    if (Object == NULL)
        return STATUS_INVALID_PARAMETER;

    if (IsEqualGUID(Guid, IID_IUnknown) ||
        IsEqualGUID(Guid, IID_IAdapterPnpManagement)) {
        *Object = (PVOID)this;
        AddRef();
        return STATUS_SUCCESS;
    }
    *Object = NULL;
    return STATUS_NOT_SUPPORTED;
}

PC_REBALANCE_TYPE CAdapter::GetSupportedRebalanceType()
{
    return PcRebalanceNotSupported;
}
VOID CAdapter::PnpQueryStop()  { }
VOID CAdapter::PnpCancelStop() { }
VOID CAdapter::PnpStop()       { }
#pragma code_seg()

// ---------------------------------------------------------------------------
// AddDevice / StartDevice.
// ---------------------------------------------------------------------------

#pragma code_seg("PAGE")
NTSTATUS StartDevice(_In_ PDEVICE_OBJECT DeviceObject, _In_ PIRP Irp,
                     _In_ PRESOURCELIST ResourceList);

NTSTATUS
AddDevice
(
    _In_ PDRIVER_OBJECT  DriverObject,
    _In_ PDEVICE_OBJECT  DeviceObject
)
{
    NTSTATUS status;
    UNREFERENCED_PARAMETER(DriverObject);

    status = PcAddAdapterDevice(DriverObject, DeviceObject, StartDevice,
                                MAX_MINIPORTS, 0);
    return status;
}

NTSTATUS
StartDevice
(
    _In_ PDEVICE_OBJECT DeviceObject,
    _In_ PIRP          Irp,
    _In_ PRESOURCELIST ResourceList
)
{
    PPORT                Port = NULL;
    PPORTTOPOLOGY        Port2Topo;
    PPORTWAVERT          Port2Wave;
    PMINIPORTTOPOLOGY    MiniportTopology = NULL;
    PMINIPORTWAVERT      MiniportWaveRT   = NULL;
    NTSTATUS             status;
    IAdapterPnpManagement* pAdapter = NULL;

    UNREFERENCED_PARAMETER(Irp);
    UNREFERENCED_PARAMETER(ResourceList);

    // ---- topology port + miniport ----
    status = PcNewPort(&Port, CLSID_PortTopology);
    if (!NT_SUCCESS(status)) {
        return status;
    }
    Port2Topo = (PPORTTOPOLOGY)Port;

    status = CMiniportTopology::Create(NULL, NULL, Port2Topo, &MiniportTopology);
    if (!NT_SUCCESS(status)) {
        Port->Release();
        return status;
    }
    status = PcRegisterSubdevice(DeviceObject, MIMETYPE_TOPOLOGY_ID,
                                 (IUnknown*)MiniportTopology);
    Port->Release();
    Port = NULL;
    if (!NT_SUCCESS(status)) {
        return status;
    }

    // ---- WaveRT port + capture miniport ----
    status = PcNewPort(&Port, CLSID_PortWaveRT);
    if (!NT_SUCCESS(status)) {
        return status;
    }
    Port2Wave = (PPORTWAVERT)Port;

    status = CMiniportWaveRT::Create(NULL, NULL, Port2Wave, &MiniportWaveRT);
    if (!NT_SUCCESS(status)) {
        Port->Release();
        return status;
    }
    status = PcRegisterSubdevice(DeviceObject, MIMETYPE_WAVE_PORT_ID,
                                 (IUnknown*)MiniportWaveRT);
    Port->Release();
    Port = NULL;
    if (!NT_SUCCESS(status)) {
        return status;
    }

    // ---- adapter PnP management ----
    status = CAdapter::Create(DeviceObject, &pAdapter);
    if (NT_SUCCESS(status)) {
        status = PcRegisterAdapterPnpManagement((IUnknown*)pAdapter,
                                                DeviceObject);
    }
    if (!NT_SUCCESS(status)) {
        return status;
    }

    // ---- private control device (broker IPC) ----
    TvmicInitControlStrings();
    status = CreateControlDevice(DeviceObject->DriverObject, DeviceObject,
                                 &g_TvmicInterfaceGuid, TVMIC_SDDL);
    if (!NT_SUCCESS(status)) {
        // Non-fatal: the endpoint still enumerates without the broker pipe.
        status = STATUS_SUCCESS;
    }

    return status;
}
#pragma code_seg()

// ---------------------------------------------------------------------------
// DriverEntry.
// ---------------------------------------------------------------------------

#pragma code_seg("INIT")
NTSTATUS
DriverEntry
(
    _In_ PDRIVER_OBJECT  DriverObject,
    _In_ PUNICODE_STRING RegistryPath
)
{
    return PcInitializeAdapterDriver(DriverObject, RegistryPath, AddDevice);
}
#pragma code_seg()
