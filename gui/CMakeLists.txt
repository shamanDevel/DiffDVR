cmake_minimum_required(VERSION 3.10)

if (${RENDERER_BUILD_GUI})
	# your configuration
	set(GUI_FILES
		main.cpp
		utils.h
		visualizer.h
		visualizer.cpp
		tf_editor.h
		tf_editor.cpp
		visualizer_kernels.h
		visualizer_kernels.cu
		quad_drawer.h
		quad_drawer.cpp
		camera_gui.h
		camera_gui.cpp
		background_worker.h

		imgui/imconfig.h
		imgui/imgui.h
		imgui/imgui.cpp
		imgui/imgui_demo.cpp
		imgui/imgui_draw.cpp
		imgui/imgui_internal.h
		imgui/imgui_widgets.cpp
		imgui/imstb_rectpack.h
		imgui/imstb_textedit.h
		imgui/imstb_truetype.h
		imgui/imgui_impl_opengl3.cpp
		imgui/imgui_impl_opengl3.h
		imgui/imgui_impl_glfw.cpp
		imgui/imgui_impl_glfw.h
		imgui/IconsFontAwesome5.h
		imgui/imgui_extension.h
		imgui/imgui_extension.cpp

		../third-party/nlohmann/json.hpp
		../third-party/lodepng/lodepng.cpp
		../third-party/lodepng/lodepng.h
		../third-party/portable-file-dialogs/portable-file-dialogs.h
		../third-party/tinyxml2/tinyxml2.h
		../third-party/tinyxml2/tinyxml2.cpp
		)
	set(GUI_NAME Gui CACHE INTERNAL "target name for the GUI")

	# create the executable
	add_executable(${GUI_NAME}
		${GUI_FILES}
		)
	target_include_directories(${GUI_NAME}
		PRIVATE 
		${LIBRARY_INCLUDE_DIR} 
		${GLM_INCLUDE_DIRS} 
		${GLEW_INCLUDE_DIR} 
		${GLFW_INCLUDE_DIRS}
		${TORCH_INCLUDE_DIR}
		${CMAKE_SOURCE_DIR}/third-party/cuMat
		${CMAKE_SOURCE_DIR}/third-party/nlohmann
		${CMAKE_SOURCE_DIR}/third-party/lodepng
		${CMAKE_SOURCE_DIR}/third-party/thread-pool/include
		${CMAKE_SOURCE_DIR}/third-party/portable-file-dialogs
		${CMAKE_SOURCE_DIR}/third-party/magic_enum/include
		${CMAKE_SOURCE_DIR}/third-party/tinyformat
		${CMAKE_SOURCE_DIR}/third-party/tinyxml2
		${CMAKE_SOURCE_DIR}/third-party/indicators/include
		)
	target_link_libraries(${GUI_NAME}
		${LIBRARY_NAME} 
		${OPENGL_LIBRARY} 
		${GLEW_SHARED_LIBRARY_RELEASE} 
		${GLFW_LIBRARIES}
		${TORCH_LIBRARIES})
	set_property(TARGET ${GUI_NAME} PROPERTY CXX_STANDARD 17)
	set_property(TARGET ${GUI_NAME} PROPERTY CUDA_ARCHITECTURES 61 72)
	add_custom_command(TARGET ${GUI_NAME}
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/bin
		COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${GUI_NAME}> ${CMAKE_SOURCE_DIR}/bin/${GUI_NAME}${CMAKE_EXECUTABLE_SUFFIX}
		
		COMMENT "Copies the executable to bin/"
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/..
		
		VERBATIM
		)
	target_compile_definitions(${GUI_NAME} 
		PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLEW=1)
	set_property(TARGET ${GUI_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
	#set_property(TARGET ${GUI_NAME} PROPERTY
	#    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

	# Include resources in executable
	file(DOWNLOAD "https://raw.githubusercontent.com/vector-of-bool/cmrc/master/CMakeRC.cmake"
					 "${CMAKE_BINARY_DIR}/CMakeRC.cmake")
	include("${CMAKE_BINARY_DIR}/CMakeRC.cmake")
	cmrc_add_resource_library(
		${GUI_NAME}-resources

		ALIAS ${GUI_NAME}::rc
		NAMESPACE ui

		resources/fa-regular-400.ttf
		resources/fa-solid-900.ttf
		)
	target_link_libraries(${GUI_NAME} ${GUI_NAME}::rc)
	#set_property(TARGET ${GUI_NAME}-resources PROPERTY
	#	MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
	set_property(TARGET ${GUI_NAME}-resources PROPERTY CUDA_ARCHITECTURES 61 72)
endif()