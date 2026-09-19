/*--

Module Name:

    minip.h

Abstract:

    Miniport declarations for the Toustovač Clean Microphone: the topology
    miniport and the capture WaveRT miniport, modelled on Microsoft SysVAD
    (see native/virtual_mic/NOTICE for the pinned upstream commit). Bodies
    live in minip.cpp and minwavert.cpp.

--*/

#pragma once

#include <windows.h>
#include <ks.h>
#include <ksmedia.h>
#include <portcls.h>
#include <ntstrsafe.h>

#include "formats.h"
#include "pcm_ring.h"
#include "public/toustovac_virtual_mic_ioctl.h"

#define MAX_NUMBER_OF_PINS     1
#define MAX_NUMBER_OF_STREAMS  1
#define TVMIC_RING_CAPACITY_FRAMES 50   /* 50 * 10 ms = 500 ms cap */

class CMiniportWaveRT;
typedef CMiniportWaveRT *PCMiniportWaveRT;

extern CPCMRing* g_pRing;

// ---------------------------------------------------------------------------
// Property table entry for the KS property dispatch.
// ---------------------------------------------------------------------------

typedef struct _TVMIC_PROPERTY_ITEM {
    ULONG nProperty;
    ULONG nItems;
    LPCVOID pData;
} TVMIC_PROPERTY_ITEM;

// ---------------------------------------------------------------------------
// CPortInfo - KS property dispatch against static tables (see minip.cpp).
// ---------------------------------------------------------------------------

class CPortInfo
{
public:
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
        _Out_ PLONG   pnMin,
        _Out_ PLONG   pnMax,
        _Out_ PMPI32  pStep
    );

protected:
    PNPAUDIO_DEVICE_CONTEXT m_Device;       // the 1st one is the dev object itself
    ULONG                   m_nDevices;
    PPCRANGES_INFORMATION   m_pRangesInfo;  // set from Init
    ULONG                   m_MinipNameSize;
    PMINIPORT_DESCRIPTOR    m_pMinipDescriptors; // adapter topology descriptor
};

// ---------------------------------------------------------------------------
// CMiniportTopology
// ---------------------------------------------------------------------------

class CMiniportTopology : public IMiniportTopology, public CPortInfo
{
public:
    static NTSTATUS Create(_In_ PNPAUDIO_DEVICE_CONTEXT DevCtx,
                           _In_ ULONG MaxNameLen,
                           _In_ PMINIPORT_TOPOLOGY_DESCRIPTOR Descriptor);

    ~CMiniportTopology();

    /* IUnknown */
    ULONG AddRef();
    ULONG Release();

    /* IMiniport */
    NTSTATUS Init(_In_ PNPAUDIO_DEVICE_CONTEXT DevCtx,
                  _In_ ULONG MaxNameLen,
                  _In_ PMINIPORT_TOPOLOGY_DESCRIPTOR Descriptor);
    VOID GetDescription(_Out_ PWSTR* ppwName);
    VOID SetPowerState(_In_ POWER_STATE state);
    NTSTATUS InitRanges();

    /* IMiniportTopology */
    NTSTATUS GetProperty(_In_ PNGUID pPropertySet,
                         _In_ ULONG nPropId,
                         _In_ ULONG nPropLen,
                         _Out_writes_bytes_to_opt_(nPropLen, *PNPropLen) PVOID pProp,
                         _Out_opt_ PULONG PNPropLen);
    NTSTATUS GetPropertyRange(_In_ PNGUID pPropertySet,
                              _In_ ULONG nPropId,
                              _Out_ PLONG pnMin,
                              _Out_ PLONG pnMax,
                              _Out_ PMPI32 pStep);

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
    static NTSTATUS Create(_In_ PNPAUDIO_DEVICE_CONTEXT DevCtx,
                           _In_ ULONG MaxNameLen,
                           _In_ PWCHAR PnpInterface);

    ~CMiniportWaveRT();

    /* IUnknown */
    ULONG AddRef();
    ULONG Release();

    /* IMiniport */
    NTSTATUS Init(_In_ PNPAUDIO_DEVICE_CONTEXT DevCtx,
                  _In_ ULONG MaxNameLen,
                  _In_ PWCHAR PnpInterface);
    VOID GetDescription(_Out_ PWSTR* ppwName);
    VOID SetPowerState(_In_ POWER_STATE state);
    NTSTATUS InitRanges();

    /* IMiniportWaveRT */
    NTSTATUS CreateStream(_In_ ULONG nStream,
                          _In_ PVOID pPhysicalDevice,
                          _In_ PMINIPORT_PROPERTY pProperty,
                          _Out_ IMiniportWaveRTStream** PpStream);

    LONG GetRef() const { return m_RefCount; }

    // Shared with the control device (broker) + the stream DPC.
    CPCMRing* m_pRing;
};
