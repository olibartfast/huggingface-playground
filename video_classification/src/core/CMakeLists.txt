add_library(video_classification_core STATIC
    json_utils.cpp
    triton_client.cpp
    video_processor.cpp
    image_processor.cpp
    videomae_image_processor.cpp
    vivit_image_processor.cpp
    timesformer_image_processor.cpp
    video_utils.cpp
)

target_include_directories(video_classification_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Triton client library configuration
# Set TRITON_CLIENT_ROOT via environment variable or CMake cache
if(NOT DEFINED TRITON_CLIENT_ROOT)
    if(DEFINED ENV{TRITON_CLIENT_ROOT})
        set(TRITON_CLIENT_ROOT $ENV{TRITON_CLIENT_ROOT})
    else()
        set(TRITON_CLIENT_ROOT "/home/oli/dependencies/triton_client_libs/install" CACHE PATH "Path to Triton client library installation")
    endif()
endif()

target_include_directories(video_classification_core SYSTEM PUBLIC
    ${TRITON_CLIENT_ROOT}/include
)

find_library(TRITON_HTTP_CLIENT httpclient PATHS ${TRITON_CLIENT_ROOT}/lib NO_DEFAULT_PATH)
if(NOT TRITON_HTTP_CLIENT)
    message(FATAL_ERROR "httpclient library not found in ${TRITON_CLIENT_ROOT}/lib")
endif()

target_link_libraries(video_classification_core PUBLIC
    fmt::fmt
    rapidjson
    ${OpenCV_LIBS}
    ${TRITON_HTTP_CLIENT}
)

target_link_directories(video_classification_core PUBLIC
    ${TRITON_CLIENT_ROOT}/lib
)





# Enforce C++20
target_compile_features(video_classification_core PUBLIC cxx_std_20)

# Add warnings
target_link_libraries(video_classification_core PRIVATE project_warnings)
