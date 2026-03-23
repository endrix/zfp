#include "cuZFP.h"
#include "../gpu_backend.h"

extern "C" size_t
zfp_gpu_compress(zfp_stream* stream, const zfp_field* field)
{
  return cuda_compress(stream, field);
}

extern "C" void
zfp_gpu_decompress(zfp_stream* stream, zfp_field* field)
{
  cuda_decompress(stream, field);
}
