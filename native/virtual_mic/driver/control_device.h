/*--

Module Name:

    control_device.h

Abstract:

    Creation and dispatch of the private control device object used by
    ToustovacAudioBroker.exe. One handle, one packet header layout
    (TvmicPacketV1, see public/toustovac_virtual_mic_ioctl.h).

--*/

#pragma once

#include <ntddk.h>
#include <wdmsec.h>
#include "public/toustovac_virtual_mic_ioctl.h"

// Creation of the control device + device interface.
NTSTATUS
CreateControlDevice
(
    _In_ PDRIVER_OBJECT     DriverObject,
    _In_ PDEVICE_OBJECT     PhysicalDeviceObject,
    _In_ LPCGUID            pInterfaceGuid,
    _In_ PCUNICODE_STRING   pSddl
);

// IRP_MJ_DEVICE_CONTROL dispatcher for the IOCTLs above.
VOID
TvmicCtlDispatch
(
    _In_ PDEVICE_OBJECT DeviceObject,
    _In_ PIRP          Irp
);

// Initialize the module-level names/SDDL strings prior to device creation.
VOID
TvmicInitControlStrings
(
    VOID
);

// Module-level state (defined in control_device.cpp).
extern UNICODE_STRING  g_TvmicControlName;
extern UNICODE_STRING  g_TvmicSddl;
extern GUID            g_TvmicInterfaceGuid;
extern UNICODE_STRING  g_TvmicInterfaceRef;

#define TVMIC_SDDL &g_TvmicSddl
