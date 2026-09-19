/* pch.h — forced first include for all engine translation units.
 *
 * <winsock2.h> must precede <windows.h> so that winsock.h (pulled in by
 * windows.h) is suppressed via the _WINSOCKAPI_ guard: otherwise the
 * shared/um socket structs are redefined and the build breaks.
 */
#ifndef JARVIS_AUDIO_ENGINE_PCH_H_
#define JARVIS_AUDIO_ENGINE_PCH_H_

#ifndef NOMINMAX
#define NOMINMAX
#endif

/* M_PI / M_SQRT2 for math.h under MSVC */
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

/* skip mmsystem/wincrypt/etc. (X509_NAME & friends are wincrypt macros that
 * collide with the generated proto identifiers) */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <winsock2.h>
#include <windows.h>
#include <ws2tcpip.h>
#include <mmeapi.h>

/* 1) lowercase legacy names must behave as plain identifiers in WebRTC's C
 *    sources; 2) uppercase FAR/NEAR must expand to NOTHING for mmreg.h's
 *    `typedef X NEAR *P;` lines; 3) wingdi/sal value macros clash with the
 *    generated proto identifiers; 4) `interface` keyword-macro for the SDK. */
#undef far
#undef near
#undef small
#define FAR
#define NEAR
#undef ERROR
#undef TRANSPARENT
#undef interface
#define interface struct

/* 256-bit vector ABI for the *_avx2.cc kernels (their operator[] on
 * __m128/__m256 only exists under the vector ABI). */
#define __AVX__
#define __AVX2__

#endif /* JARVIS_AUDIO_ENGINE_PCH_H_ */
