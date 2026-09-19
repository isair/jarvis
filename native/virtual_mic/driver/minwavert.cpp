/*--
    minwavert.cpp - module globals + WaveRT table helpers (see minip.h).
--*/

#include "minwavert.h"

// The module owns exactly one shared ring and one WaveRT instance.
PCMiniportWaveRT  g_pWaveRt;
CPCMRing*         g_pRing;
