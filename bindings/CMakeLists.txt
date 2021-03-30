cmake_minimum_required(VERSION 3.10)

# your configuration
set(TEST_FILES
	bindings.cpp
	)
set(BINDINGS_NAME pyrenderer CACHE INTERNAL "target name for the bindings")

# create the python extension
pybind11_add_module(${BINDINGS_NAME}
	${TEST_FILES}
	)
target_include_directories(${BINDINGS_NAME}
	PRIVATE 
	${LIBRARY_INCLUDE_DIR} 
	${GLM_INCLUDE_DIRS} 
	${TORCH_INCLUDE_DIR}
	${CMAKE_SOURCE_DIR}/third-party/tinyformat
	${CMAKE_SOURCE_DIR}/third-party/cuMat
	${CMAKE_SOURCE_DIR}/third-party/cuMat/third-party
	${CMAKE_SOURCE_DIR}/third-party/cudad/include/cudAD
	${CMAKE_SOURCE_DIR}/third-party/nlohmann
	)
target_link_libraries(${BINDINGS_NAME}
	PRIVATE ${LIBRARY_NAME} 
	PRIVATE ${LIBRARY_LINK_LIBRARIES})
set_property(TARGET ${BINDINGS_NAME} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${BINDINGS_NAME} PROPERTY CUDA_ARCHITECTURES 61 72)

add_custom_command(TARGET ${BINDINGS_NAME}
	POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/bin
	COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${BINDINGS_NAME}> ${CMAKE_SOURCE_DIR}/bin/${PYTHON_MODULE_PREFIX}${BINDINGS_NAME}${PYTHON_MODULE_EXTENSION}
	
	COMMENT "Copies the python extension to bin/"
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/..
	
	VERBATIM
	)
set_property(TARGET ${BINDINGS_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

#set_property(TARGET ${BINDINGS_NAME} PROPERTY
#	MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# resources
target_link_libraries(${BINDINGS_NAME} PRIVATE ${LIBRARY_NAME}::rc)
