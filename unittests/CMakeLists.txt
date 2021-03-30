cmake_minimum_required(VERSION 3.10)

if (${RENDERER_BUILD_TESTS})

# your configuration
set(TEST_FILES
	main.cpp
	check_adjoint.h
	test_utils.h
	testAdjointUtils.cpp
	testAdjointBlending.cpp
	testAdjointTF.cpp
	testAdjointCamera.cpp
	testAdjointInterpolation.cpp
	testCudaCompilation.cpp
	testConsistencyCamera.cpp
	testConsistencyTF.cpp
	testNonConvexDensity.cpp
	testMath.cpp

	../third-party/tinyformat/tinyformat.h
	../third-party/nlohmann/json.hpp
	../third-party/lodepng/lodepng.cpp
	../third-party/lodepng/lodepng.h
	../third-party/catch/catch.hpp
	)
set(TEST_NAME Test CACHE INTERNAL "target name for the tests")

# create the executable
add_executable(${TEST_NAME}
	${TEST_FILES}
	)
target_include_directories(${TEST_NAME}
	PRIVATE 
	${LIBRARY_INCLUDE_DIR} 
	${GLM_INCLUDE_DIRS} 
	${GLEW_INCLUDE_DIR} 
	${GLFW_INCLUDE_DIRS}
	${TORCH_INCLUDE_DIR}
	${CMAKE_SOURCE_DIR}/third-party/tinyformat
	${CMAKE_SOURCE_DIR}/third-party/cuMat
	${CMAKE_SOURCE_DIR}/third-party/cuMat/third-party
	${CMAKE_SOURCE_DIR}/third-party/nlohmann
	${CMAKE_SOURCE_DIR}/third-party/lodepng
	${CMAKE_SOURCE_DIR}/third-party/thread-pool/include
	${CMAKE_SOURCE_DIR}/third-party/cudad/include/cudAD
	${CMAKE_SOURCE_DIR}/third-party/catch
	${CMAKE_SOURCE_DIR}/third-party/indicators/include
	${CMAKE_SOURCE_DIR}/third-party/magic_enum/include
	)
target_link_libraries(${TEST_NAME}
	${LIBRARY_NAME} 
	${CUDA_LIBRARIES}
	${OPENGL_LIBRARY} 
	${GLEW_SHARED_LIBRARY_RELEASE} 
	${GLFW_LIBRARIES})
set_property(TARGET ${TEST_NAME} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${TEST_NAME} PROPERTY CUDA_ARCHITECTURES 61 72)
target_compile_definitions(${TEST_NAME} PRIVATE CUMAT_EIGEN_SUPPORT=1)
add_custom_command(TARGET ${TEST_NAME}
	POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/bin
	COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TEST_NAME}> ${CMAKE_SOURCE_DIR}/bin/${TEST_NAME}${CMAKE_EXECUTABLE_SUFFIX}
	
	COMMENT "Copies the executable to bin/"
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/..
	
	VERBATIM
	)
set_property(TARGET ${TEST_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

#set_property(TARGET ${TEST_NAME} PROPERTY
#	MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

endif(${RENDERER_BUILD_TESTS})
