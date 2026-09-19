// pch_gen.h - forced first include for the generated perfetto proto TUs
// (protos / per directory *.gen.cc) and the protozero runtime.
// Plain C++ TUs: Windows keyword macros are dropped so the generated
// identifiers (interface, ERROR, far, ...) stay intact.
#ifndef JARVIS_AUDIO_ENGINE_PCH_GEN_H_
#define JARVIS_AUDIO_ENGINE_PCH_GEN_H_

#include <stddef.h>
#include <stdint.h>

#ifdef far
#undef far
#endif
#ifdef near
#undef near
#endif
#ifdef small
#undef small
#endif
#ifdef ERROR
#undef ERROR
#endif
#ifdef TRANSPARENT
#undef TRANSPARENT
#endif
#ifdef interface
#undef interface
#endif

#endif // JARVIS_AUDIO_ENGINE_PCH_GEN_H_
