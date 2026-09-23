/*--
    control_device.cpp - the private control device for the broker.

    One device object + symbolic link. Dispatch:
      IOCTL_TVMIC_NEGOTIATE      : validate canonical format + protocol v1.
      IOCTL_TVMIC_BEGIN_GENERATION / END_GENERATION : generation scope.
      IOCTL_TVMIC_WRITE_FRAMES   : validate the versioned packet header,
                                   CRC32C, sequence and size bounds, then
                                   push PCM16 into the PCM ring.
      IOCTL_TVMIC_SET_MUTE       : 0/1.
      IOCTL_TVMIC_QUERY_STATUS   : counters + last packet identity.

    Security: the SDDL grants the interface only to SYSTEM, Administrators
    and the service SID NT SERVICE\ToustovacAudioBroker.
--*/

#include "control_device.h"
#include "minip.h"

#include <ks.h>
#include <ksmedia.h>

// ---------------------------------------------------------------------------
// Module-level names/SDDL.
// ---------------------------------------------------------------------------

UNICODE_STRING g_TvmicControlName;
UNICODE_STRING g_TvmicSddl;
GUID           g_TvmicInterfaceGuid = GUID_DEVINTERFACE_TOUSTOVAC_VIRTUAL_MIC_CONTROL;
UNICODE_STRING g_TvmicInterfaceRef;

static const WCHAR s_TvmicControlName[] = L"\\Device\\ToustovacCleanMic";
static const WCHAR s_TvmicSddlText[] =
    L"D:"
    L"(A;;GA;;;SY)"          /* SYSTEM */
    L"(A;;GA;;;BA)"          /* Administrators */
    L"(A;;GA;;;S-1-5-80-2524399536-2920122170-3629737132-2493692265-2421734816)";
                                /* NT SERVICE\ToustovacAudioBroker */

VOID
TvmicInitControlStrings(VOID)
{
    RtlInitUnicodeString(&g_TvmicControlName, s_TvmicControlName);
    RtlInitUnicodeString(&g_TvmicSddl, s_TvmicSddlText);
}

// ---------------------------------------------------------------------------
// Control-side state. All shared audio data lives in the CPCMRing owned by
// the WaveRT miniport; the control side only holds scalar counters.
// ---------------------------------------------------------------------------

static CPCMRing* g_pCtlRing = NULL;
static LONG       g_Muted = 0;
static LONG       g_RejectedPackets = 0;
static LONG       g_ProtocolVersion = 0;
static ULONGLONG  g_Generation = 0;

// CRC32C (Castagnoli, reflected poly 0x82F63B78).
static ULONG TvmicCrc32c(const UCHAR* pData, ULONG nLength)
{
    static ULONG table[256];
    static BOOLEAN built = FALSE;
    ULONG crc = 0xFFFFFFFF;
    ULONG i;
    int j;

    if (!built) {
        for (i = 0; i < 256; i++) {
            ULONG v = i;
            for (j = 0; j < 8; j++)
                v = (v & 1) ? (v >> 1) ^ 0x82F63B78 : v >> 1;
            table[i] = v;
        }
        built = TRUE;
    }
    for (i = 0; i < nLength; i++)
        crc = table[(crc ^ pData[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFF;
}

// ---------------------------------------------------------------------------
// Device creation.
// ---------------------------------------------------------------------------

#pragma code_seg("PAGE")
NTSTATUS
CreateControlDevice
(
    _In_ PDRIVER_OBJECT   DriverObject,
    _In_ PDEVICE_OBJECT   PhysicalDeviceObject,
    _In_ LPCGUID          pInterfaceGuid,
    _In_ PCUNICODE_STRING pSddl
)
{
    NTSTATUS        status;
    PDEVICE_OBJECT  controlObject = NULL;

    UNREFERENCED_PARAMETER(pSddl);

    status = IoCreateDeviceSecure(
        DriverObject,
        0,
        &g_TvmicControlName,
        FILE_DEVICE_UNKNOWN,
        FILE_DEVICE_SECURE_OPEN,
        FALSE,
        &g_TvmicSddl,
        pInterfaceGuid,
        &controlObject
    );
    if (!NT_SUCCESS(status)) {
        return status;
    }

    status = IoRegisterDeviceInterface(PhysicalDeviceObject, pInterfaceGuid,
                                       NULL, &g_TvmicInterfaceRef);
    if (!NT_SUCCESS(status)) {
        IoDeleteDevice(controlObject);
        return status;
    }
    status = IoSetDeviceInterfaceState(&g_TvmicInterfaceRef, TRUE);
    if (!NT_SUCCESS(status)) {
        IoDeleteDevice(controlObject);
        return status;
    }
    return STATUS_SUCCESS;
}
#pragma code_seg()

// ---------------------------------------------------------------------------
// IRP_MJ_DEVICE_CONTROL dispatcher.
// ---------------------------------------------------------------------------

#pragma code_seg("PAGE")
VOID
TvmicCtlDispatch
(
    _In_ PDEVICE_OBJECT DeviceObject,
    _In_ PIRP          Irp
)
{
    PIO_STACK_LOCATION  irpSp = IoGetCurrentIrpStackLocation(Irp);
    NTSTATUS            status = STATUS_SUCCESS;
    ULONG               bytesReturned = 0;
    ULONG               inLen;
    ULONG               outLen;
    PUCHAR              inBuf;
    PUCHAR              outBuf;

    UNREFERENCED_PARAMETER(DeviceObject);

    inLen = irpSp->Parameters.DeviceIoControl.InputBufferLength;
    outLen = irpSp->Parameters.DeviceIoControl.OutputBufferLength;
    inBuf = (PUCHAR)Irp->AssociatedIrp.SystemBuffer;
    outBuf = (PUCHAR)Irp->AssociatedIrp.SystemBuffer;

    switch (irpSp->Parameters.DeviceIoControl.IoControlCode) {

    case IOCTL_TVMIC_NEGOTIATE:
    {
        PTvmicInit pInit;
        PTvmicInit pOut;
        if (inBuf == NULL || inLen < sizeof(TvmicInit) ||
            outBuf == NULL || outLen < sizeof(TvmicInit)) {
            status = STATUS_INVALID_PARAMETER;
            break;
        }
        pInit = (PTvmicInit)inBuf;
        if (pInit->struct_size != sizeof(TvmicInit) ||
            pInit->protocol_version != TVMIC_PROTOCOL_VERSION ||
            pInit->sample_rate != TVMIC_SAMPLE_RATE ||
            pInit->channels != TVMIC_CHANNELS ||
            pInit->bits_per_sample != TVMIC_BITS_PER_SAMPLE ||
            pInit->frame_samples != TVMIC_FRAME_SAMPLES) {
            status = STATUS_REVISION_MISMATCH;
            break;
        }
        g_pCtlRing = (g_pWaveRt != NULL) ? g_pWaveRt->m_pRing : NULL;
        if (g_pCtlRing == NULL) {
            status = STATUS_DEVICE_POWERED_OFF;
            break;
        }
        g_pCtlRing->Initialize(g_pCtlRing->CapacityFrames());
        g_Generation = pInit->producer_generation;
        g_ProtocolVersion = 1;

        pOut = (PTvmicInit)outBuf;
        *pOut = *pInit;
        bytesReturned = sizeof(TvmicInit);
        break;
    }

    case IOCTL_TVMIC_BEGIN_GENERATION:
    {
        if (g_pCtlRing == NULL) { status = STATUS_DEVICE_POWERED_OFF; break; }
        if (inLen != sizeof(ULONGLONG) || inBuf == NULL) {
            status = STATUS_INVALID_PARAMETER; break;
        }
        /* fresh generation => drop stale buffered PCM atomically */
        g_pCtlRing->Reset();
        g_Generation = *(PULONGLONG)inBuf;
        bytesReturned = 0;
        break;
    }

    case IOCTL_TVMIC_END_GENERATION:
    {
        if (g_pCtlRing == NULL) { status = STATUS_DEVICE_POWERED_OFF; break; }
        g_pCtlRing->Reset();
        bytesReturned = 0;
        break;
    }

    case IOCTL_TVMIC_WRITE_FRAMES:
    {
        PTvmicPacketV1 pHdr;
        ULONG nSamples;
        ULONG pcmBytes;
        PUCHAR pPcm;

        if (g_pCtlRing == NULL) { status = STATUS_DEVICE_POWERED_OFF; break; }
        if (g_ProtocolVersion != (LONG)TVMIC_PROTOCOL_VERSION) {
            status = STATUS_REVISION_MISMATCH; break;
        }
        if (inBuf == NULL || inLen < sizeof(TvmicPacketV1)) {
            status = STATUS_INVALID_PARAMETER; break;
        }
        pHdr = (PTvmicPacketV1)inBuf;
        nSamples = pHdr->sample_count;
        pcmBytes = nSamples * 2u;

        /* bounded batch, no overflow, whole 10 ms frames */
        if (pHdr->struct_size != (UINT32)sizeof(TvmicPacketV1) ||
            pHdr->protocol_version != TVMIC_PROTOCOL_VERSION ||
            nSamples == 0 ||
            pcmBytes > 2u * 1024u ||
            inLen != (ULONG)sizeof(TvmicPacketV1) + pcmBytes ||
            (nSamples % TVMIC_FRAME_SAMPLES) != 0) {
            g_RejectedPackets++;
            status = STATUS_INVALID_PARAMETER;
            break;
        }
        if (pHdr->sample_rate != TVMIC_SAMPLE_RATE ||
            pHdr->channels != TVMIC_CHANNELS ||
            pHdr->bits_per_sample != TVMIC_BITS_PER_SAMPLE) {
            g_RejectedPackets++;
            status = STATUS_REVISION_MISMATCH;
            break;
        }
        pPcm = inBuf + sizeof(TvmicPacketV1);
        if (TvmicCrc32c(pPcm, pcmBytes) != pHdr->payload_crc32c) {
            g_RejectedPackets++;
            status = STATUS_DATA_ERROR;
            break;
        }
        if (pHdr->producer_generation < g_Generation) {
            g_RejectedPackets++;
            status = STATUS_DATA_ERROR;
            break;
        }
        if (pHdr->producer_generation > g_Generation) {
            g_pCtlRing->Reset();
            g_Generation = pHdr->producer_generation;
        }
        if (g_pCtlRing->LastSequence() != 0 &&
            pHdr->sequence != g_pCtlRing->LastSequence() + 1 &&
            (pHdr->flags & TVMIC_PACKET_FLAG_DISCONTINUITY) == 0) {
            g_RejectedPackets++;
            status = STATUS_DATA_ERROR;
            break;
        }
        if (g_Muted) {
            INT16 silence[TVMIC_FRAME_SAMPLES];
            RtlZeroMemory(silence, sizeof(silence));
            g_pCtlRing->Write(silence, TVMIC_FRAME_SAMPLES,
                              pHdr->qpc_100ns, pHdr->flags);
        } else {
            g_pCtlRing->Write((const INT16*)pPcm, nSamples,
                              pHdr->qpc_100ns, pHdr->flags);
        }
        bytesReturned = 0;
        break;
    }

    case IOCTL_TVMIC_SET_MUTE:
    {
        if (inLen != sizeof(LONG) || inBuf == NULL) {
            status = STATUS_INVALID_PARAMETER; break;
        }
        g_Muted = *(PLONG)inBuf ? 1 : 0;
        bytesReturned = 0;
        break;
    }

    case IOCTL_TVMIC_QUERY_STATUS:
    {
        PTvmicCtlStatus pSt;
        if (outBuf == NULL || outLen < sizeof(TvmicCtlStatus)) {
            status = STATUS_INVALID_PARAMETER; break;
        }
        pSt = (PTvmicCtlStatus)outBuf;
        RtlZeroMemory(pSt, sizeof(*pSt));
        pSt->struct_size = sizeof(TvmicCtlStatus);
        pSt->protocol = (UINT32)g_ProtocolVersion;
        pSt->producer_generation = g_Generation;
        if (g_pCtlRing != NULL) {
            pSt->active_capture_clients = 1;
            pSt->ring_capacity_frames = g_pCtlRing->CapacityFrames();
            pSt->ring_depth_frames = g_pCtlRing->Depth();
            pSt->last_sequence = g_pCtlRing->LastSequence();
            pSt->last_timestamp_100ns = g_pCtlRing->LastQpc();
            pSt->frames_produced = g_pCtlRing->FramesProduced();
            pSt->silence_frames = g_pCtlRing->SilenceFrames();
            pSt->driver_underflows = g_pCtlRing->Underruns();
            pSt->driver_overflows = g_pCtlRing->FramesDropped();
            pSt->stale_packets = g_pCtlRing->StalePackets();
            pSt->sequence_gaps = g_pCtlRing->SequenceGaps();
            pSt->max_ring_depth_frames = g_pCtlRing->MaxDepth();
        }
        pSt->rejected_packets = (UINT64)g_RejectedPackets;
        bytesReturned = sizeof(TvmicCtlStatus);
        break;
    }

    default:
        status = STATUS_INVALID_DEVICE_REQUEST;
        break;
    }

    Irp->IoStatus.Status = status;
    Irp->IoStatus.Information = bytesReturned;
    IoCompleteRequest(Irp, IO_NO_INCREMENT);
}
#pragma code_seg()
