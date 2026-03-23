set(MLX_HINT_ROOT "${MLX_DIR}" CACHE PATH "Path to MLX build/install root")

if(NOT DEFINED MLX_HINT_ROOT OR MLX_HINT_ROOT STREQUAL "")
  message(FATAL_ERROR "ZFP_WITH_MLX requires MLX_DIR to be set")
endif()

if(EXISTS "${MLX_HINT_ROOT}/libmlx.a")
  set(_mlx_lib "${MLX_HINT_ROOT}/libmlx.a")
  get_filename_component(_mlx_repo_root "${MLX_HINT_ROOT}" DIRECTORY)
elseif(EXISTS "${MLX_HINT_ROOT}/lib/libmlx.a")
  set(_mlx_lib "${MLX_HINT_ROOT}/lib/libmlx.a")
  set(_mlx_repo_root "${MLX_HINT_ROOT}")
else()
  message(FATAL_ERROR "Could not find libmlx.a under ${MLX_HINT_ROOT}")
endif()

if(NOT TARGET mlx)
  add_library(mlx STATIC IMPORTED)
  set_target_properties(mlx PROPERTIES IMPORTED_LOCATION "${_mlx_lib}")
endif()

set(MLX_INCLUDE_DIRS "${_mlx_repo_root}")

if(EXISTS "${MLX_HINT_ROOT}/_deps/metal_cpp-src")
  list(APPEND MLX_INCLUDE_DIRS "${MLX_HINT_ROOT}/_deps/metal_cpp-src")
endif()

add_library(zfp_mlx STATIC
  mlx_zfp.cpp)

target_compile_features(zfp_mlx PRIVATE cxx_std_17)

target_include_directories(zfp_mlx
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ZFP_SOURCE_DIR}/include
    ${MLX_INCLUDE_DIRS})

target_link_libraries(zfp_mlx
  PRIVATE
    zfp
    mlx
    "-framework Metal"
    "-framework Foundation"
    "-framework QuartzCore"
    "-framework Accelerate")

if(BUILD_TESTING OR BUILD_TESTING_FULL)
  add_executable(testMlxZfp test_mlx_zfp.cpp)
  target_compile_features(testMlxZfp PRIVATE cxx_std_17)
  target_include_directories(testMlxZfp
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${ZFP_SOURCE_DIR}/include
      ${MLX_INCLUDE_DIRS})
  target_link_libraries(testMlxZfp
    PRIVATE
      zfp_mlx
      zfp
      mlx
      "-framework Metal"
      "-framework Foundation"
      "-framework QuartzCore"
      "-framework Accelerate")
  add_test(NAME testMlxZfp COMMAND testMlxZfp)
endif()
