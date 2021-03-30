cmake_minimum_required(VERSION 3.16)

find_package(OpenMP)

# OUTPUT CACHE VARIABLES:
# BINDING_NAME
# LIBRARY_INCLUDE_DIR
# LIBRARY_LINK_LIBRARIES

# your configuraton
set(LIBRARY_HEADER_FILES
	commons.h
	lib.h
	errors.h
	volume.h
	renderer.h
	renderer_cpu.h
	renderer_cuda.h
	settings.h
	camera.h
	halton_sampler.h
	tf_utils.h
	sha1.h
	pytorch_utils.h
	losses.h
	kernel_loader.h
	)
set(LIBRARY_CUDA_HEADER_FILES
	helper_math.cuh
	renderer_commons.cuh
	renderer_tensor.cuh
	renderer_settings.cuh
	renderer_utils.cuh
	renderer_interpolation.cuh
	renderer_tf.cuh
	renderer_blending.cuh
	renderer_camera.cuh
	renderer_kernels.cuh
	renderer_compareToImage.cuh
	renderer_adjoint.cuh
	cooperative_groups.cuh
	)
set(LIBRARY_CUDA_FILES
	renderer_cuda_impl.cu
	camera_impl.cu
	losses_impl.cu
	tf_utils_impl.cu
	#volume.cu
	)
set(LIBRARY_SOURCE_FILES
	volume.cpp
	volume_implicit.cpp
	camera.cpp
	tf_utils.cpp
	renderer.cpp
	renderer_cpu.cpp
	renderer_cuda.cpp
	sha1.cpp
	losses.cpp
	kernel_loader.cpp
	)
# the target name of the library
set(LIBRARY_NAME Renderer CACHE INTERNAL "the target name of the library, also used for the binding")

#add_library(${LIBRARY_NAME}_cuda OBJECT ${LIBRARY_CUDA_FILES})

# get list of filenames
file(GLOB RENDERER_KERNEL_FILES1
	list_directories false
	${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)
file(GLOB RENDERER_KERNEL_FILES2
	list_directories false
	${CMAKE_SOURCE_DIR}/third-party/cudad/include/cudAD/*.cuh
	${CMAKE_SOURCE_DIR}/third-party/cudad/include/cudAD/*.h)
set(RENDERER_KERNEL_FILES
	${RENDERER_KERNEL_FILES1} ${RENDERER_KERNEL_FILES2}
	PARENT_SCOPE)

if(NOT ${RENDERER_ONLY_RESOURCES})

    # the library, compiled as a shared library
    set(LIBRARY_ALL_FILES
        ${LIBRARY_HEADER_FILES}
        ${LIBRARY_CUDA_HEADER_FILES}
        ${LIBRARY_SOURCE_FILES}
        ${LIBRARY_CUDA_FILES}
    )
    if(RENDERER_SHARED_LIB)
        add_library(${LIBRARY_NAME} SHARED
            ${LIBRARY_ALL_FILES})
    else(RENDERER_SHARED_LIB)
        add_library(${LIBRARY_NAME} STATIC
            ${LIBRARY_ALL_FILES})
    endif(RENDERER_SHARED_LIB)
    source_group("CUDA Headers" REGULAR_EXPRESSION ".*\.cuh")

    set_property(TARGET ${LIBRARY_NAME} PROPERTY CUDA_ARCHITECTURES 61 72)
    set_target_properties(${LIBRARY_NAME} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    if (WIN32)
        target_compile_options(${LIBRARY_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${MY_CUDA_NVCC_FLAGS}>)
    else()
        target_compile_options(${LIBRARY_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17 --expt-relaxed-constexpr --extended-lambda>)
    endif()

    # openmp
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${LIBRARY_NAME} optimized OpenMP::OpenMP_CXX)
    endif()

    # windows: bigobj
    if (WIN32)
        set_property(SOURCE renderer_cpu.cpp APPEND PROPERTY COMPILE_OPTIONS "/bigobj")
    endif()
    #set_property(TARGET ${LIBRARY_NAME} PROPERTY
	#    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

    # get include path to the first library header, to be used in the test application
    list(GET ${LIBRARY_HEADER_FILES} 0 LIBRARY_INCLUDE_DIRb)
    get_filename_component(LIBRARY_INCLUDE_DIRa ${LIBRARY_INCLUDE_DIRb} ABSOLUTE)
    get_filename_component(LIBRARY_INCLUDE_DIRc ${LIBRARY_INCLUDE_DIRa} DIRECTORY)
    set(LIBRARY_INCLUDE_DIR ${LIBRARY_INCLUDE_DIRc} CACHE FILEPATH "include directory of the custom library") 
    target_include_directories(${LIBRARY_NAME}
        PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${PYTHON_INCLUDE_DIR} 
            ${GLM_INCLUDE_DIRS}
            ${TORCH_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/third-party/cuMat
            ${CMAKE_SOURCE_DIR}/third-party/magic_enum/include
            ${CMAKE_SOURCE_DIR}/third-party/cudad/include/cudAD
            ${CMAKE_SOURCE_DIR}/third-party/tinyformat
        INTERFACE ${LIBRARY_INCLUDE_DIR}
        )
    # libraries, also used in the test application
    set(LIBRARY_LINK_LIBRARIES
        ${PYTHON_LIBRARY}
        CACHE STRING "python and torch libraries")
    target_link_libraries(${LIBRARY_NAME}
        ${LIBRARY_LINK_LIBRARIES}
        ${CUDA_LIBRARIES}
        CUDA::cuda_driver CUDA::nvrtc CUDA::curand
        ${TORCH_LIBRARIES}
        lz4)
    set_property(TARGET ${LIBRARY_NAME} PROPERTY CXX_STANDARD 17)
    set_property(TARGET ${LIBRARY_NAME} PROPERTY CUDA_STANDARD 17)

    # CUDA runtime compilation
    list(GET ARCH_FLAGS -1 RENDERER_NVCC_FLAGS)
    set(RENDERER_KERNEL_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/third-party/cudad/include/cudAD)
    set(RENDERER_KERNEL_DIRS_STR "{")
    foreach(line IN LISTS RENDERER_KERNEL_DIRS)
        string(APPEND RENDERER_KERNEL_DIRS_STR "\"${line}\",")
    endforeach() 
    string(APPEND RENDERER_KERNEL_DIRS_STR "}")
    message(STATUS "renderer kernel directories: " ${RENDERER_KERNEL_DIRS_STR})
    if (WIN32)
        target_compile_definitions(${LIBRARY_NAME} PUBLIC 
            RENDERER_SHADER_DIRS=${RENDERER_KERNEL_DIRS_STR}
            RENDERER_RUNTIME_COMPILATION=1
            NVCC_ARGS="${RENDERER_NVCC_FLAGS}"
            NVCC_INCLUDE_DIR=${CUDA_INCLUDE_DIRS})
    else()
        # there are some errors that RENDERER_SHADER_DIRS is not properly passed to the compiler
        # therefore, it is required on Unix to provide a custom source lodaer
        target_compile_definitions(${LIBRARY_NAME} PUBLIC 
            RENDERER_RUNTIME_COMPILATION=1
            NVCC_ARGS="${RENDERER_NVCC_FLAGS}")
    endif()

    # other options
    if (RENDERER_BUILD_CPU_KERNELS)
        target_compile_definitions(${LIBRARY_NAME} PUBLIC 
            RENDERER_BUILD_CPU_KERNELS=1)
    else()
        target_compile_definitions(${LIBRARY_NAME} PUBLIC 
            RENDERER_BUILD_CPU_KERNELS=0)
    endif()

    # Shared or static
    if(RENDERER_SHARED_LIB)
        target_compile_definitions(${LIBRARY_NAME}
            PUBLIC RENDERER_BUILD_SHARED 
            PRIVATE BUILD_MAIN_LIB)
    else(RENDERER_SHARED_LIB)
        # no extra definitions
    endif(RENDERER_SHARED_LIB)

    if(RENDERER_SHARED_LIB)
    # post-build commands, copies the dll to the bin/ folder
    get_filename_component(LIBRARY_INCLUDE_DIR ${LIBRARY_INCLUDE_DIRa} DIRECTORY)
    add_custom_command(TARGET ${LIBRARY_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/bin
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${LIBRARY_NAME}> ${CMAKE_SOURCE_DIR}/bin/${LIBRARY_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}

        COMMENT "Copies the libarary .dll to bin/"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/..
        
        VERBATIM
        )
    endif(RENDERER_SHARED_LIB)

    # Dark Magic to fix "nvcc fatal: A single input file is required for a non-link phase when an outputfile is specified"
    #set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )

    # debugging: keep .ptx files of the kernels
    #set_property(TARGET ${LIBRARY_NAME}_cuda PROPERTY CUDA_PTX_COMPILATION ON)
    #target_compile_options(${LIBRARY_NAME} PRIVATE "--keep")

endif()
