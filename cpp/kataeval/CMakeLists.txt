# kataeval — the clean "KataGo neural-net evaluation" dependency, as one target.
#
#   add_subdirectory(kataeval)
#   target_link_libraries(your_app kataeval)   # + Dawn (webgpu_dawn) + ZLIB
#
# The contents are the minimal KataGo subset in sources.txt (single source of
# truth, shared with scripts/build-eval.sh). Public interface: kataeval.h.
cmake_minimum_required(VERSION 3.18)

# Read the manifest; resolve paths relative to cpp/ (this dir's parent).
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/sources.txt" _lines)
set(KATAEVAL_SOURCES "")
foreach(_l ${_lines})
  string(STRIP "${_l}" _l)
  if(_l AND NOT _l MATCHES "^#")
    list(APPEND KATAEVAL_SOURCES "${CMAKE_CURRENT_LIST_DIR}/../${_l}")
  endif()
endforeach()

add_library(kataeval STATIC ${KATAEVAL_SOURCES})
set_target_properties(kataeval PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_include_directories(kataeval PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(kataeval PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../external
  ${CMAKE_CURRENT_LIST_DIR}/../external/filesystem-1.5.8/include)
target_compile_definitions(kataeval PRIVATE USE_WEBGPU_BACKEND HALF_ENABLE_CPP11_CFENV=0)

# The WebGPU backend TU needs C++20 (webgpu_cpp.h); the rest is C++17.
set_source_files_properties(
  "${CMAKE_CURRENT_LIST_DIR}/../neuralnet/webgpubackend.cpp"
  PROPERTIES COMPILE_OPTIONS "-std=c++20")

# Consumers must provide the two external runtime deps:
#   - Dawn / webgpu_dawn  (native WebGPU)   — target dawn::webgpu_dawn
#   - ZLIB                (gzip model read) — find_package(ZLIB)
# e.g.  target_link_libraries(kataeval PUBLIC dawn::webgpu_dawn ZLIB::ZLIB)
