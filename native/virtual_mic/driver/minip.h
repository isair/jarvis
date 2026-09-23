/*--

Module Name:

    minip.h

Abstract:

    Miniport declarations for the Toustovač Clean Microphone: the topology
    miniport and the capture WaveRT miniport, modelled on Microsoft SysVAD
    (see native/virtual_mic/NOTICE for the pinned upstream commit). Bodies
    live in minip.cpp and minwavert.cpp.

    Modern PortCls (WDM portcls.h): miniports are created in StartDevice and
    registered with PcRegisterSubdevice; property dispatch uses the
    PCPROPERTY_ITEM / PCPFNPROPERTY_HANDLER model.

--*/

#pragma once

#define INITGUID
#include <ntddk.h>
#include <portcls.h>
#include <stdunk.h>
#include <ks.h>
#include <ksmedia.h>
#include <ntstrsafe.h>

#include "formats.h"
#include "pcm_ring.h"
#include "public/toustovac_virtual_mic_ioctl.h"

#define MAX_NUMBER_OF_PINS     1
#define MAX_NUMBER_OF_STREAMS  1
#define TVMIC_RING_CAPACITY_FRAMES 50   /* 50 * 10 ms = 500 ms cap */

class CMiniportWaveRT;
typedef CMiniportWaveRT *PCMiniportWaveRT;

extern CPCMRing*        g_pRing;
extern PCMiniportWaveRT g_pWaveRt;

// ---------------------------------------------------------------------------
// CPortInfo - KS property dispatch against static tables (see minip.cpp).
// ---------------------------------------------------------------------------

class CPortInfo
{
public:
    NTSTATUS
    GetProperty
    (
        _In_ PPCPROPERTY_REQUEST PropertyRequest
    );

    NTSTATUS
    GetPropertyRange
    (
        _In_ PPCPROPERTY_REQUEST PropertyRequest
    );

protected:
    PDEVICE_OBJECT            m_Device;           // the 1st one is the dev object itself
    ULONG                     m_nDevices;
    PPCSTREAMRESOURCE_DESCRIPTOR m_pRangesInfo;   // set from Init
    ULONG                     m_MinipNameSize;
    PCM_PARTIAL_RESOURCE_DESCRIPTOR m_pMinipDescriptors; // adapter topo descriptor
};

// ---------------------------------------------------------------------------
// CMiniportTopology
// ---------------------------------------------------------------------------

class CMiniportTopology : public IMiniportTopology, public CPortInfo
{
public:
    static NTSTATUS Create(_In_ PUNKNOWN UnknownAdapter,
                           _In_ PRESOURCELIST ResourceList,
                           _In_ PPORTTOPOLOGY Port,
                           _Outptr_ PMINIPORTTOPOLOGY *Miniport);

    ~CMiniportTopology();

    /* IUnknown */
    NTSTATUS QueryInterface(_In_ REFGUID Guid, _Outptr_ PVOID *Object);
    ULONG AddRef();
    ULONG Release();

    /* IMiniport */
    NTSTATUS Init(_In_ PUNKNOWN UnknownAdapter,
                  _In_ PRESOURCELIST ResourceList,
                  _In_ PPORTTOPOLOGY Port);
    NTSTATUS GetDescription(_Out_ PPCFILTER_DESCRIPTOR *ppwDescription);
    NTSTATUS DataRangeIntersection(_In_ ULONG PinId,
                                   _In_ PKSDATARANGE DataRange,
                                   _In_ PKSDATARANGE MatchingDataRange,
                                   _In_ ULONG OutputBufferLength,
                                   _Out_writes_bytes_to_opt_(OutputBufferLength, *ResultantFormatLength) PVOID ResultantFormat,
                                   _Out_ PULONG ResultantFormatLength);
    VOID SetPowerState(_In_ POWER_STATE state);

    /* IMiniportTopology */
    NTSTATUS GetProperty(_In_ PPCPROPERTY_REQUEST PropertyRequest);
    NTSTATUS GetPropertyRange(_In_ PPCPROPERTY_REQUEST PropertyRequest);

protected:
    LONG                    m_RefCount;
    PWSTR                   m_pwMyName;      // device-friendly name
    ULONG                   m_MaxNameLen;    // in chars w/o NUL
    LONG                    m_nRanges;
};

// ---------------------------------------------------------------------------
// CMiniportWaveRT
// ---------------------------------------------------------------------------

class CMiniportWaveRT : public IMiniportWaveRT, public CPortInfo
{
public:
    static NTSTATUS Create(_In_ PUNKNOWN UnknownAdapter,
                           _In_ PRESOURCELIST ResourceList,
                           _In_ PPORTWAVERT Port,
                           _Outptr_ PMINIPORTWAVERT *Miniport);

    ~CMiniportWaveRT();

    /* IUnknown */
    NTSTATUS QueryInterface(_In_ REFGUID Guid, _Outptr_ PVOID *Object);
    ULONG AddRef();
    ULONG Release();

    /* IMiniport */
    NTSTATUS Init(_In_ PUNKNOWN UnknownAdapter,
                  _In_ PRESOURCELIST ResourceList,
                  _In_ PPORTWAVERT Port);
    NTSTATUS GetDescription(_Out_ PPCFILTER_DESCRIPTOR *ppwDescription);
    NTSTATUS DataRangeIntersection(_In_ ULONG PinId,
                                   _In_ PKSDATARANGE DataRange,
                                   _In_ PKSDATARANGE MatchingDataRange,
                                   _In_ ULONG OutputBufferLength,
                                   _Out_writes_bytes_to_opt_(OutputBufferLength, *ResultantFormatLength) PVOID ResultantFormat,
                                   _Out_ PULONG ResultantFormatLength);

    /* IMiniportWaveRT */
    NTSTATUS NewStream(_Out_ PMINIPORTWAVERTSTREAM *Stream,
                       _In_ PPORTWAVERTSTREAM PortStream,
                       _In_ ULONG Pin,
                       _In_ BOOLEAN Capture,
                       _In_ PKSDATAFORMAT DataFormat);
    NTSTATUS GetDeviceDescription(_Out_ PDEVICE_DESCRIPTION DeviceDescription);
    VOID SetPowerState(_In_ POWER_STATE state);

    LONG GetRef() const { return m_RefCount; }

    // Shared with the control device (broker) + the stream DPC.
    CPCMRing* m_pRing;

protected:
    LONG                    m_RefCount;
    PWSTR                   m_pwMyName;
    ULONG                   m_MaxNameLen;
    LONG                    m_nRanges;
};
