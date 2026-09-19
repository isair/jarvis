/*--
    formats.h - Endpoint formats for the Toustovač Clean Microphone. One
    canonical 48 kHz mono 16-bit PCM format (480 frames / 10 ms), matching
    the CleanAudioBus and the named-pipe wire format exactly.
--*/

#pragma once

#include <portcls.h>
#include <ks.h>
#include <ksmedia.h>

// 48 kHz, mono, 16-bit PCM.
static const WORD NUM_SPEC_48_MONO = 48;
static const WORD NUM_SPEC_48_16_MONO = 192;

/* Default mix format: 48000 Hz, mono, 16 bits, block align 2. */
static const PCM16_WAVE_FORMAT WaveFormat48kMono =
{
    {
        WAVE_FORMAT_PCM,     /* wFormatTag */
        1,                   /* nChannels */
        48000,               /* nSamplesPerSec */
        96000,               /* nAvgBytesPerSec */
        2,                   /* nBlockAlign */
        16,                  /* wBitsPerSample */
        0                    /* cbSize */
    },
    {(4+8+16-16), 0, 0, 0},  /* channel mask: mono */
};

/* The KS data range for the single canonical format: 480-sample frames. */
static const KS_DATARANGE 48K_Range = { 48000, NUM_SPEC_48_MONO };
