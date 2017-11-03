# CMake entry point.
# Note that this CMakeLists.txt in fully independent from the one in the parent directory
# (since that's the goal of this tutorial !)

# For an introduction to CMake, see
# http://www.cmake.org/cmake/help/cmake_tutorial.html (at least steps 1 and 2)

cmake_minimum_required (VERSION 3.1)
set(name "YOLO")
set(exename "yolo_exe")
set(libname "yolo_lib")
project ( ${name} )

#c++ version
set (CMAKE_CXX_STANDARD 11)

#opencv 
if ( WIN32 )
	set(OpenCV_DIR "E:/libs/OpenCV/x64/vc14/lib")
	set(OpenCV_CONFIG_PATH "E:/libs/OpenCV")
	set(WIN_HEADER_PATH "${PROJECT_SOURCE_DIR}/winheadfile")
    include_directories(${WIN_HEADER_PATH})
endif()

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#cuda
find_package(CUDA)
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
#    -O3 
	-gencode=arch=compute_61,code=\"sm_61,compute_61\"
    )

set(MY_HEADER_FILES
	# C++ header file
    activations.h
    activation_layer.h
    avgpool_layer.h
    batchnorm_layer.h
    blas.h
    box.h
    classifier.h
    col2im.h
    connected_layer.h
    convolutional_layer.h
    cost_layer.h
    crnn_layer.h
    crop_layer.h
    cuda.h
    data.h
    deconvolutional_layer.h
    demo.h
    detection_layer.h
    dropout_layer.h
    gemm.h
    getopt.h
    gettimeofday.h
    gru_layer.h
    im2col.h
    image.h
    layer.h
    list.h
    local_layer.h
    matrix.h
    maxpool_layer.h
    network.h
    normalization_layer.h
    option_list.h
    parser.h
    region_layer.h
    reorg_layer.h
    rnn_layer.h
    route_layer.h
    shortcut_layer.h
    softmax_layer.h
    stb_image.h
    stb_image_write.h
    tree.h
    unistd.h
    utils.h
)
set(MY_CUDA_FILES
    # CUDA files
    activation_kernels.cu
    avgpool_layer_kernels.cu
    blas_kernels.cu
    col2im_kernels.cu
    convolutional_kernels.cu
    crop_layer_kernels.cu
    deconvolutional_kernels.cu
    dropout_layer_kernels.cu
    im2col_kernels.cu
    maxpool_layer_kernels.cu
    network_kernels.cu
)
set(MY_SOURCE_FILES 
	# C++ code
    activations.c
    activation_layer.c
    art.c
    avgpool_layer.c
    batchnorm_layer.c
    blas.c
    box.c
    captcha.c
    cifar.c
    classifier.c
    coco.c
    col2im.c
    compare.c
    connected_layer.c
    convolutional_layer.c
    cost_layer.c
    cpu_gemm.c
    crnn_layer.c
    crop_layer.c
    cuda.c
    darknet.c
    data.c
    deconvolutional_layer.c
    demo.c
    detection_layer.c
    detector.c
    dice.c
    dropout_layer.c
    gemm.c
    getopt.c
    gettimeofday.c
    go.c
    gru_layer.c
    im2col.c
    image.c
    layer.c
    list.c
    local_layer.c
    matrix.c
    maxpool_layer.c
    network.c
    nightmare.c
    normalization_layer.c
    option_list.c
    parser.c
    region_layer.c
    reorg_layer.c
    rnn.c
    rnn_layer.c
    rnn_vid.c
    route_layer.c
    shortcut_layer.c
    softmax_layer.c
    super.c
    swag.c
    tag.c
    tree.c
    utils.c
    voxel.c
    writing.c
    yolo.c
    yolo_v2_class.cpp
    yolo_v2_class.hpp
)

include_directories(
    E:/libs/YOLO/include
    E:/libs/cuDNN/include
)
set(YOLO_LIBS
    E:/libs/YOLO/lib/pthreadVC2.lib
    E:/libs/cuDNN/lib/x64/cudnn.lib
)
cuda_add_executable(${exename}
    ${MY_CUDA_FILES}
	${MY_HEADER_FILES}
    ${MY_SOURCE_FILES}
    yolo_console_dll.cpp
)

add_definitions(-DYOLODLL_EXPORTS)
add_definitions(-DGPU)

cuda_add_library(${libname} SHARED
    ${MY_CUDA_FILES}
    ${MY_HEADER_FILES}
    ${MY_SOURCE_FILES}
    OPTIONS -arch sm_61
)

target_link_libraries(${exename}
    ${OpenCV_LIBS}
    ${YOLO_LIBS} 
	${CUDA_LIBRARIES}
	${CUDA_CUBLAS_LIBRARIES}
	${CUDA_curand_LIBRARY}
	${CUDA_npp_LIBRARY}
	${CUDA_nppc_LIBRARY}
)

target_link_libraries(${libname} 
    ${OpenCV_LIBS}
    ${YOLO_LIBS} 
	${CUDA_LIBRARIES}
	${CUDA_CUBLAS_LIBRARIES}
	${CUDA_curand_LIBRARY}
	${CUDA_npp_LIBRARY}
	${CUDA_nppc_LIBRARY}
)
