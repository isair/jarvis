/*--
    minwavert.h - shared module-level globals and creation helpers.
--*/

#pragma once

#include "minip.h"

// One WaveRT instance and one shared PCM ring exist per device.
PCMiniportWaveRT  g_pWaveRt     = NULL;
CPCMRing*         g_pRing       = NULL;
