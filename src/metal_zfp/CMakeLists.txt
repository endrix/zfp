set(_zfp_metal_backend_obj
  metal_zfp/metal_backend.mm)

if(ZFP_WITH_METAL_NATIVE)
  set(ZFP_METAL_AIR_FILE ${CMAKE_CURRENT_BINARY_DIR}/pack_unpack.air)
  set(ZFP_METAL_LIB_FILE ${CMAKE_CURRENT_BINARY_DIR}/pack_unpack.metallib)

  add_custom_command(
    OUTPUT ${ZFP_METAL_AIR_FILE}
    COMMAND xcrun -sdk macosx metal -O3 -c ${CMAKE_CURRENT_SOURCE_DIR}/pack_unpack.metal -o ${ZFP_METAL_AIR_FILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/pack_unpack.metal
    COMMENT "Compiling Metal source to AIR"
    VERBATIM)

  add_custom_command(
    OUTPUT ${ZFP_METAL_LIB_FILE}
    COMMAND xcrun -sdk macosx metallib ${ZFP_METAL_AIR_FILE} -o ${ZFP_METAL_LIB_FILE}
    DEPENDS ${ZFP_METAL_AIR_FILE}
    COMMENT "Linking AIR to metallib"
    VERBATIM)

  add_custom_target(zfp_metal_metallib ALL DEPENDS ${ZFP_METAL_LIB_FILE})

  list(APPEND _zfp_metal_backend_obj metal_zfp/metal_runtime.mm)
  set(_zfp_metal_native_defs
    ZFP_WITH_METAL_NATIVE
    ZFP_METAL_LIB_PATH="${ZFP_METAL_LIB_FILE}")

  if(ZFP_WITH_METAL_CODEC_EXPERIMENTAL)
    list(APPEND _zfp_metal_native_defs ZFP_WITH_METAL_CODEC_EXPERIMENTAL)
  endif()

  set(zfp_metal_native_defs
    ${_zfp_metal_native_defs}
    PARENT_SCOPE)

  set(zfp_metal_native_targets zfp_metal_metallib PARENT_SCOPE)
endif()

set(zfp_metal_backend_obj
  ${_zfp_metal_backend_obj}
  PARENT_SCOPE)
