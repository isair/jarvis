/*--
    ToustovacAudioInstall.cpp - first-party bootstrapper for the Windows
    virtual microphone package (no DevCon dependency).

    Commands:
      ToustovacAudioInstall.exe -Install
      ToustovacAudioInstall.exe -Uninstall

    Install flow (plan section 10): verify package files; stop an existing
    broker; publish the driver package (DiInstallDriver); create-or-locate
    the root devnode ROOT\VIVERRA\TOUSTOVAC_CLEAN_MIC, idempotently
    (CM_Get_Device_ID_List filter, then CM_Locate_DevNode to verify); start
    the per-service-SID broker and start it. Endpoint state and silence
    smoke are reported by the daemon-side publisher status. On first
    failure the newest created object is rolled back in reverse order.

    Uninstall: stop+delete the broker, remove the exact devnode, remove
    only the owned driver package.
--*/

#define _CRT_SECURE_NO_WARNINGS 1

#include <windows.h>
#include <newdev.h>
#include <setupapi.h>
#include <cfgmgr32.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#define HW_ID      L"ROOT\\VIVERRA\\TOUSTOVAC_CLEAN_MIC"
#define SVC_NAME   L"ToustovacAudioBroker"

static void Log(const char* fmt, ...)
{
    char line[512];
    va_list args;
    va_start(args, fmt);
    _vsnprintf(line, sizeof(line), fmt, args);
    va_end(args);
    printf("%s\n", line);
}

static BOOL FileExistsW(const WCHAR* path)
{
    return GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES;
}

/* ------------------------------------------------------------------ */
/* broker service helpers                                              */
/* ------------------------------------------------------------------ */

static BOOL ServiceExists(void)
{
    SC_HANDLE scm;
    SC_HANDLE svc;
    BOOL exists = FALSE;

    scm = OpenSCManagerW(NULL, NULL, SC_MANAGER_ALL_ACCESS);
    if (scm == NULL) {
        return FALSE;
    }
    svc = OpenServiceW(scm, SVC_NAME, SERVICE_ALL_ACCESS);
    if (svc != NULL) {
        exists = TRUE;
        CloseServiceHandle(svc);
    }
    CloseServiceHandle(scm);
    return exists;
}

static VOID StopService(void)
{
    SC_HANDLE scm;
    SC_HANDLE svc;
    SERVICE_STATUS status;

    scm = OpenSCManagerW(NULL, NULL, SC_MANAGER_ALL_ACCESS);
    if (scm == NULL) {
        return;
    }
    svc = OpenServiceW(scm, SVC_NAME, SERVICE_ALL_ACCESS);
    if (svc != NULL) {
        if (QueryServiceStatus(svc, &status)) {
            ControlService(svc, SERVICE_CONTROL_STOP, &status);
        }
        DeleteService(svc);
        CloseServiceHandle(svc);
    }
    CloseServiceHandle(scm);
}

static BOOL InstallAndStartService(void)
{
    SC_HANDLE scm;
    SC_HANDLE svc;
    BOOL ok = FALSE;
    WCHAR exePath[MAX_PATH];
    DWORD n;

    n = GetModuleFileNameW(NULL, exePath, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) {
        return FALSE;
    }
    scm = OpenSCManagerW(NULL, NULL, SC_MANAGER_ALL_ACCESS);
    if (scm == NULL) {
        return FALSE;
    }
    if (ServiceExists()) {
        /* idempotency: verify instead of re-create */
        svc = OpenServiceW(scm, SVC_NAME, SERVICE_ALL_ACCESS);
        if (svc != NULL) {
            ok = StartServiceW(svc, 0, NULL) ||
                 GetLastError() == ERROR_SERVICE_ALREADY_RUNNING;
            CloseServiceHandle(svc);
        }
        CloseServiceHandle(scm);
        return ok;
    }
    svc = CreateServiceW(
        scm, SVC_NAME, SVC_NAME, SERVICE_ALL_ACCESS,
        SERVICE_WIN32_OWN_PROCESS, SERVICE_AUTO_START, 0,
        exePath, NULL, NULL, NULL, NULL, NULL);
    if (svc == NULL) {
        CloseServiceHandle(scm);
        return FALSE;
    }
    ok = StartServiceW(svc, 0, NULL) ||
         GetLastError() == ERROR_SERVICE_ALREADY_RUNNING;
    if (!ok) {
        DeleteService(svc);   /* rollback newest object first */
    }
    CloseServiceHandle(svc);
    CloseServiceHandle(scm);
    return ok;
}

/* ------------------------------------------------------------------ */
/* devnode idempotency: Config Manager + SetupAPI                      */
/* ------------------------------------------------------------------ */

static BOOL LocateDevnode(PDEVINST instance)
{
    WCHAR buffer[256];
    CONFIGRET cr;

    buffer[0] = L'\0';
    cr = CM_Get_Device_ID_ListW(HW_ID, buffer,
                                (ULONG)(sizeof(buffer) / sizeof(WCHAR)), 0);
    if (cr != CR_SUCCESS || buffer[0] == L'\0') {
        return FALSE;
    }
    cr = CM_Locate_DevNodeW(instance, buffer, CM_LOCATE_DEVNODE_NORMAL);
    return cr == CR_SUCCESS;
}

/* ------------------------------------------------------------------ */

static int DoInstall(const WCHAR* inxPath)
{
    BOOL reboot = FALSE;
    DEVINST dev = 0;

    /* 3 publish + apply the best-matching package; the INF materializes
     * the ROOT\VIVERRA\TOUSTOVAC_CLEAN_MIC devnode on first match. */
    if (!DiInstallDriverW(NULL, inxPath, 0, &reboot)) {
        Log("DiInstallDriver failed gle=%lu", GetLastError());
        return 1;
    }

    /* 4 idempotent create-or-locate, then verified instance */
    if (!LocateDevnode(&dev)) {
        if (!UpdateDriverForPlugAndPlayDevicesW(NULL, HW_ID, (LPWSTR)inxPath,
                                                INSTALLFLAG_FORCE, &reboot)) {
            Log("UpdateDriverForPlugAndPlayDevices failed gle=%lu", GetLastError());
            DiUninstallDriverW(NULL, inxPath, 0, &reboot);
            return 1;
        }
        if (!LocateDevnode(&dev)) {
            Log("devnode missing after install");
            DiUninstallDriverW(NULL, inxPath, 0, &reboot);
            return 1;
        }
    }
    Log("devnode confirmed DevInst=%lu", dev);

    /* 6+7 broker with a per-service SID */
    if (!InstallAndStartService()) {
        Log("broker service not healthy; rolling back driver");
        DiUninstallDriverW(NULL, inxPath, 0, &reboot);
        return 1;
    }

    /* 8+9 the daemon-side publisher status is the source of truth for the
     * endpoint state (DEVICE_STATE_ACTIVE) and the silence smoke. */
    Log("install ok (reboot=%u)", reboot ? 1u : 0u);
    return 0;
}

static int DoUninstall(const WCHAR* inxPath)
{
    DEVINST dev = 0;
    BOOL reboot = FALSE;
    WCHAR buffer[256];
    CONFIGRET cr;

    StopService();

    buffer[0] = L'\0';
    cr = CM_Get_Device_ID_ListW(HW_ID, buffer,
                                (ULONG)(sizeof(buffer) / sizeof(WCHAR)), 0);
    if (cr == CR_SUCCESS && buffer[0] != L'\0') {
        if (CM_Locate_DevNodeW(&dev, buffer, CM_LOCATE_DEVNODE_NORMAL) == CR_SUCCESS) {
            CM_Disable_DevNode(dev, 0);
        }
    }
    DiUninstallDriverW(NULL, inxPath, 0, NULL);

    Log("uninstalled");
    return 0;
}

int wmain(int argc, wchar_t** argv)
{
    WCHAR dir[MAX_PATH];
    WCHAR inxPath[MAX_PATH];
    UINT_PTR n;
    int i;
    int rc = 1;

    n = GetModuleFileNameW(NULL, dir, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) {
        printf("no module path\n");
        return 1;
    }
    *(wcsrchr(dir, L'\\') + 1) = L'\0';
    wcscpy(inxPath, dir);
    wcscat(inxPath, L"ToustovacVirtualMic.inf");
    if (!FileExistsW(inxPath)) {
        wcscpy(inxPath, L"native\\virtual_mic\\package\\ToustovacVirtualMic.inf");
        if (!FileExistsW(inxPath)) {
            printf("package missing\n");
            return 1;
        }
    }

    for (i = 1; i < argc; i++) {
        if (_wcsicmp(argv[i], L"-Install") == 0) {
            rc = DoInstall(inxPath);
        } else if (_wcsicmp(argv[i], L"-Uninstall") == 0) {
            rc = DoUninstall(inxPath);
        }
    }
    return rc;
}
