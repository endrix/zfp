#include "zfp.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>
#include <string.h>

struct setupVars {
  zfp_stream* stream;
  zfp_field* field;
  bitstream* bs;
  void* buffer;
  float* data;
  size_t streamSize;
};

static int
setup(void **state)
{
  struct setupVars *bundle = malloc(sizeof(struct setupVars));
  assert_non_null(bundle);

  bundle->stream = zfp_stream_open(NULL);
  assert_non_null(bundle->stream);

  bundle->buffer = malloc(50 * sizeof(int));
  assert_non_null(bundle->buffer);
  memset(bundle->buffer, 0, 50 * sizeof(int));

  bundle->bs = stream_open(bundle->buffer, 50 * sizeof(int));
  zfp_stream_set_bit_stream(bundle->stream, bundle->bs);
  stream_skip(bundle->bs, stream_word_bits + 1);
  bundle->streamSize = stream_size(bundle->bs);
  assert_int_not_equal(bundle->streamSize, 0);

  assert_int_equal(1, zfp_stream_set_execution(bundle->stream, zfp_exec_cuda));

  bundle->data = malloc(22 * sizeof(float));
  assert_non_null(bundle->data);
  memset(bundle->data, 0, 22 * sizeof(float));

  bundle->field = zfp_field_1d(bundle->data, zfp_type_float, 11);
  assert_non_null(bundle->field);
  zfp_field_set_stride_1d(bundle->field, 2);

  (void)zfp_stream_set_rate(bundle->stream, 8.0, zfp_type_float, 1, zfp_false);

  *state = bundle;
  return 0;
}

static int
teardown(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field_free(bundle->field);
  stream_close(bundle->bs);
  free(bundle->data);
  free(bundle->buffer);
  zfp_stream_close(bundle->stream);
  free(bundle);
  return 0;
}

static void
given_noncontiguous_when_compress_cuda_policy_expect_noop(void **state)
{
  struct setupVars *bundle = *state;
  size_t compressedSize = zfp_compress(bundle->stream, bundle->field);
  size_t endSize = stream_size(bundle->bs);
  assert_int_equal(compressedSize, endSize);
  assert_true(endSize >= bundle->streamSize);
}

static void
given_noncontiguous_when_decompress_cuda_policy_expect_noop(void **state)
{
  struct setupVars *bundle = *state;
  assert_int_equal(zfp_decompress(bundle->stream, bundle->field), bundle->streamSize);
  assert_int_equal(stream_size(bundle->bs), bundle->streamSize);
}

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_setup_teardown(given_noncontiguous_when_compress_cuda_policy_expect_noop, setup, teardown),
    cmocka_unit_test_setup_teardown(given_noncontiguous_when_decompress_cuda_policy_expect_noop, setup, teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
