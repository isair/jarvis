/*--
    minwavert.h - shared module-level globals and creation helpers.
--*/

#pragma once

#include "minip.h"

// One WaveRT instance and one shared PCM ring exist per device.
extern PCMiniportWaveRT g_pWaveRt;
extern CPCMRing*        g_pRing;
