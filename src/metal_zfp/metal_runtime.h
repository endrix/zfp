#ifndef ZFP_METAL_RUNTIME_H
#define ZFP_METAL_RUNTIME_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bitstream bitstream;

int zfp_metal_pack_runtime(const void* src,
                                size_t src_bytes,
                                void* packed,
                                const unsigned int dims[3],
                                const ptrdiff_t stride[3],
                                long long int offset,
                                size_t item_size);

int zfp_metal_unpack_runtime(const void* packed,
                                  size_t packed_bytes,
                                  void* dst,
                                  size_t dst_bytes,
                                  const unsigned int dims[3],
                                  const ptrdiff_t stride[3],
                                  long long int offset,
                                  size_t item_size);

int zfp_metal_encode_contiguous_runtime(const void* src,
                                        size_t src_bytes,
                                        void* dst,
                                        size_t dst_bytes,
                                        size_t item_size,
                                        int dim);

int zfp_metal_decode_contiguous_runtime(const void* src,
                                        size_t src_bytes,
                                        void* dst,
                                        size_t dst_bytes,
                                        size_t item_size,
                                        int dim);

size_t zfp_metal_host_encode2d_float(void* zfp,
                                     const float* src,
                                     unsigned int nx,
                                     unsigned int ny,
                                     ptrdiff_t sx,
                                     ptrdiff_t sy,
                                     unsigned int maxbits,
                                     void* stream_words,
                                     size_t stream_capacity_bytes);

int zfp_metal_host_decode2d_float(void* zfp,
                                  const void* stream_words,
                                  size_t stream_bytes,
                                  float* dst,
                                  unsigned int nx,
                                  unsigned int ny,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  unsigned int maxbits);

size_t zfp_metal_encode1d_float_runtime(const float* src,
                                        unsigned int dim,
                                        int sx,
                                        unsigned int maxbits,
                                        void* stream_words,
                                        size_t stream_capacity_bytes);

size_t zfp_metal_decode1d_float_runtime(const void* stream_words,
                                        unsigned int dim,
                                        int sx,
                                        unsigned int maxbits,
                                        float* dst);

size_t zfp_metal_encode2d_float_runtime(const float* src,
                                        unsigned int nx,
                                        unsigned int ny,
                                        ptrdiff_t sx,
                                        ptrdiff_t sy,
                                        unsigned int maxbits,
                                        void* stream_words,
                                        size_t stream_capacity_bytes);

size_t zfp_metal_decode2d_float_runtime(const void* stream_words,
                                        unsigned int nx,
                                        unsigned int ny,
                                        ptrdiff_t sx,
                                        ptrdiff_t sy,
                                        unsigned int maxbits,
                                        float* dst);

size_t zfp_metal_encode3d_float_runtime(const float* src,
                                        unsigned int nx,
                                        unsigned int ny,
                                        unsigned int nz,
                                        ptrdiff_t sx,
                                        ptrdiff_t sy,
                                        ptrdiff_t sz,
                                        unsigned int maxbits,
                                        void* stream_words,
                                        size_t stream_capacity_bytes);

size_t zfp_metal_decode3d_float_runtime(const void* stream_words,
                                        unsigned int nx,
                                        unsigned int ny,
                                        unsigned int nz,
                                        ptrdiff_t sx,
                                        ptrdiff_t sy,
                                        ptrdiff_t sz,
                                        unsigned int maxbits,
                                        float* dst);

#ifdef __cplusplus
}
#endif

#endif
