/*--
    driver_client.cpp - sole writable handle to the driver control device.

    Resolves the private device-interface symbolic link through the SetupAPI
    and opens one plain duplicated file handle. The control interface is
    SDDL-restricted by the driver (SYSTEM, Administrators,
    NT SERVICE\ToustovacAudioBroker).
--*/

#include "broker.h"
#include "eventlog.h"

#include <stdio.h>
#include <string.h>

BOOL DriverClientOpen(BrokerContext* ctx)
{
    GUID classGuid = GUID_DEVINTERFACE_TOUSTOVAC_VIRTUAL_MIC_CONTROL;
    WCHAR detailBuf[40];
    PSP_DEVICE_INTERFACE_DETAIL_DATA_W detail;
    HDEVINFO set;
    DWORD i;
    BOOL found = FALSE;

    if (ctx == NULL) {
        return FALSE;
    }
    detailBuf[0] = L'\0';

    set = SetupDiGetClassDevsW(&classGuid, NULL, NULL,
                               DIGCF_DEVICEINTERFACE | DIGCF_PRESENT);
    if (set == INVALID_HANDLE_VALUE) {
        BrokerEventLog(EVENTLOG_ERROR_TYPE, "driver_missing: no control interface");
        return FALSE;
    }

    detail = (PSP_DEVICE_INTERFACE_DETAIL_DATA_W)detailBuf;
    detail->cbSize = (DWORD)sizeof(*detail);

    for (i = 0; !found; i++) {
        SP_DEVICE_INTERFACE_DATA iface;
        DWORD chars = 0;

        iface.cbSize = (DWORD)sizeof(iface);
        if (!SetupDiEnumDeviceInterfaces(set, NULL, &classGuid, i, &iface)) {
            break;
        }
        if (!SetupDiGetDeviceInterfaceDetailW(set, &iface, detail,
                                              sizeof(detailBuf), &chars, NULL) &&
            GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            break;
        }
        if (detail->DevicePath[0] != L'\0') {
            found = TRUE;
        }
    }
    SetupDiDestroyDeviceInfoList(set);

    if (!found) {
        BrokerEventLog(EVENTLOG_ERROR_TYPE, "driver_missing: interface not found");
        return FALSE;
    }

    ctx->Driver = CreateFileW(detail->DevicePath,
                              GENERIC_READ | GENERIC_WRITE, 0,
                              NULL, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
    if (ctx->Driver == INVALID_HANDLE_VALUE) {
        BrokerEventLog(EVENTLOG_ERROR_TYPE, "driver_missing: open failed (%lu)",
                       GetLastError());
        ctx->Driver = NULL;
        return FALSE;
    }
    return TRUE;
}

VOID DriverClientClose(BrokerContext* ctx)
{
    if (ctx != NULL && ctx->Driver != NULL && ctx->Driver != INVALID_HANDLE_VALUE) {
        CloseHandle(ctx->Driver);
    }
    if (ctx != NULL) {
        ctx->Driver = NULL;
    }
}

BOOL DriverNegotiate(BrokerContext* ctx, const TvmicInit* init, TvmicInit* out)
{
    DWORD fetched = 0;
    if (ctx == NULL || ctx->Driver == NULL) {
        return FALSE;
    }
    if (!DeviceIoControl(ctx->Driver, IOCTL_TVMIC_NEGOTIATE, (LPVOID)init,
                         sizeof(*init), out, sizeof(*out), &fetched, NULL)) {
        return FALSE;
    }
    return fetched == sizeof(*out);
}

BOOL DriverBeginGeneration(BrokerContext* ctx, ULONGLONG generation)
{
    DWORD fetched = 0;
    if (ctx == NULL || ctx->Driver == NULL) {
        return FALSE;
    }
    return DeviceIoControl(ctx->Driver, IOCTL_TVMIC_BEGIN_GENERATION,
                           &generation, sizeof(generation), NULL, 0,
                           &fetched, NULL) ? TRUE : FALSE;
}

BOOL DriverEndGeneration(BrokerContext* ctx)
{
    DWORD fetched = 0;
    if (ctx == NULL || ctx->Driver == NULL) {
        return FALSE;
    }
    return DeviceIoControl(ctx->Driver, IOCTL_TVMIC_END_GENERATION, NULL, 0,
                           NULL, 0, &fetched, NULL) ? TRUE : FALSE;
}

BOOL DriverWriteFrames(BrokerContext* ctx, const UCHAR* buf, ULONG len)
{
    DWORD fetched = 0;
    if (ctx == NULL || ctx->Driver == NULL) {
        return FALSE;
    }
    return DeviceIoControl(ctx->Driver, IOCTL_TVMIC_WRITE_FRAMES, (LPVOID)buf,
                           len, NULL, 0, &fetched, NULL) ? TRUE : FALSE;
}

BOOL DriverSetMute(BrokerContext* ctx, LONG muted)
{
    DWORD fetched = 0;
    if (ctx == NULL || ctx->Driver == NULL) {
        return FALSE;
    }
    return DeviceIoControl(ctx->Driver, IOCTL_TVMIC_SET_MUTE, &muted,
                           sizeof(muted), NULL, 0, &fetched, NULL) ? TRUE : FALSE;
}

BOOL DriverQueryStatus(BrokerContext* ctx, TvmicCtlStatus* status)
{
    DWORD fetched = 0;
    if (ctx == NULL || ctx->Driver == NULL) {
        return FALSE;
    }
    if (!DeviceIoControl(ctx->Driver, IOCTL_TVMIC_QUERY_STATUS, NULL, 0,
                         status, sizeof(*status), &fetched, NULL)) {
        return FALSE;
    }
    return fetched == sizeof(*status);
}
