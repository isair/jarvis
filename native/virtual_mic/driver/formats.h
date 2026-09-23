/*--
    formats.h - Endpoint formats for the Toustovač Clean Microphone. One
    canonical 48 kHz mono 16-bit PCM format (480 frames / 10 ms), matching
    the CleanAudioBus and the named-pipe wire format exactly.
--*/

#pragma once

#include <portcls.h>
#include <ks.h>
#include <ksmedia.h>

// 48 kHz, mono, 16-bit PCM (NUM_SPEC_* are in topology.h).

/* Default mix format: 48000 Hz, mono, 16 bits, block align 2. */
static const WAVEFORMATEXTENSIBLE WaveFormat48kMono =
{
    {
        WAVE_FORMAT_EXTENSIBLE, /* wFormatTag */
        1,                      /* nChannels  */
        48000,                  /* nSamplesPerSec */
        96000,                  /* nAvgBytesPerSec */
        2,                      /* nBlockAlign */
        16,                     /* wBitsPerSample */
        sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX) /* cbSize */
    },
    { 16 },                     /* wValidBitsPerSample */
    0,                          /* dwChannelMask (mono) */
    KSDATAFORMAT_SUBTYPE_PCM    /* SubFormat */
};
