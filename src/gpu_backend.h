#ifndef ZFP_GPU_BACKEND_H
#define ZFP_GPU_BACKEND_H

#include <stddef.h>
#include "zfp.h"

#ifdef __cplusplus
extern "C" {
#endif

size_t zfp_gpu_compress(zfp_stream* stream, const zfp_field* field);
void zfp_gpu_decompress(zfp_stream* stream, zfp_field* field);

#ifdef __cplusplus
}
#endif

#endif
