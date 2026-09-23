/*--

Module Name:

    adapter.h

Abstract:

    Adapter for the Toustovač Clean Microphone: topology + WaveRT capture
    miniports, following Microsoft SysVAD (Windows-driver-samples,
    audio/sysvad). Modern PortCls: the adapter is an IAdapterPnpManagement
    registered via PcRegisterAdapterPnpManagement; the miniports are created
    in StartDevice and registered with PcRegisterSubdevice.

--*/

#pragma once

#include <ntddk.h>
#include <portcls.h>
#include <ks.h>
#include <ksmedia.h>
#include <ntstrsafe.h>
#include "minip.h"

// Product-owned identifiers. Generated once, committed; never SysVAD's.
// Control interface (registered by the driver for the broker):
//   {A3F5D6B1-2C8E-4A7F-9B1D-6E2F0C8A4417}
// Endpoint / mic (IMMDevice):
//   {B72E94C4-1D3F-5A86-BC0A-9D7421E3F2B0}

#define MAX_MINIPORTS 2

class CAdapter : public IAdapterPnpManagement
{
public:
    CAdapter();
    static NTSTATUS Create(_Inout_ PDEVICE_OBJECT DeviceObject,
                           _Outptr_ IAdapterPnpManagement **PPAdapter);

    ~CAdapter();

    /* IUnknown */
    NTSTATUS QueryInterface(_In_ REFGUID Guid, _Outptr_ PVOID *Object);
    ULONG AddRef();
    ULONG Release();

    /* IAdapterPnpManagement */
    PC_REBALANCE_TYPE GetSupportedRebalanceType();
    VOID PnpQueryStop();
    VOID PnpCancelStop();
    VOID PnpStop();

private:
    NTSTATUS Init(_Inout_ PDEVICE_OBJECT DeviceObject);

    LONG           m_RefCount;
    PDEVICE_OBJECT m_Device;
};

// ---------------------------------------------------------------------------
// Port entry points (DriverEntry is defined in adapter.cpp too).
// ---------------------------------------------------------------------------

NTSTATUS
AdapterInitialization
(
    _Inout_ PDEVICE_OBJECT DeviceObject
);

VOID
AdapterCleanup
(
    _In_ PDRIVER_OBJECT DriverObject,
    _In_ PDEVICE_OBJECT DeviceObject
);
