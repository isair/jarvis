/*--

Module Name:

    adapter.h

Abstract:

    Adapter for the Toustovač Clean Microphone: topology + WaveRT capture
    miniports, following Microsoft SysVAD (Windows-driver-samples,
    audio/sysvad, commit 3c3fb49073c047c4cc8e6c203c6331f62b426507).

--*/

#pragma once

#include <windows.h>
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

class CAdapter : public IWaveAdapterDevice
{
public:
    static NTSTATUS Create(_Inout_ PNPAUDIO_DEVICE_CONTEXT DevCtx,
                           _In_ PADAPTER_PROTOCOL_SET ProtoSet);

    ~CAdapter();

    /* IWaveAdapterDevice */
    NTSTATUS
    GetMiniports
    (
        _Outptr_ PMINIPORT_ARRAY* Miniports,
        _Out_ PULONG NbPorts
    );

    /* IPinDriverProperty */
    NTSTATUS
    GetProperty
    (
        _In_  PNGUID  pPropertySet,
        _In_  ULONG   nPropId,
        _In_  ULONG   nPropLen,
        _Out_writes_bytes_to_opt_(nPropLen, *PNPropLen) PVOID pProp,
        _Out_opt_ PULONG PNPropLen
    );

    NTSTATUS
    GetPropertyRange
    (
        _In_  PNGUID  pPropertySet,
        _In_  ULONG   nPropId,
        _Out_ PLONG   nMin,
        _Out_ PLONG   nMax,
        _Out_ PMPI32  pStep
    );

private:
    NTSTATUS
    Init
    (
        _In_ PNPAUDIO_DEVICE_CONTEXT Device,
        _In_ PADAPTER_PROTOCOL_SET ProtoSet
    );

    LONG                    m_RefCount;
    PNPAUDIO_DEVICE_CONTEXT m_Device;         // the dev object itself
    ULONG                   m_nDevices;       // 2 for the adapter itself
    PMINIPORT_ARRAY         m_Miniports;      // one wave + one topo port
    ULONG                   m_NbMiniports;
    PPCRANGES_INFORMATION   m_pRangesInfo;
    ULONG                   m_MinipNameSize;
    PMINIPORT_DESCRIPTOR    m_pMinipDescriptors; // adapter topo descriptor
};

// ---------------------------------------------------------------------------
// Port entry points (DriverEntry is defined in adapter.cpp too).
// ---------------------------------------------------------------------------

NTSTATUS
AdapterInitialization
(
    _Inout_ PDEVICE_OBJECT DeviceObject,
    _Inout_ PADAPTER_PROTOCOL_SET ProtoSet,
    _Inout_ PADAPTER_INIT_TABLE InitTable
);

VOID
AdapterCleanup
(
    _In_ PDRIVER_OBJECT DriverObject,
    _In_ PDEVICE_OBJECT DeviceObject
);
