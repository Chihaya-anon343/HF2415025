cmake_minimum_required(VERSION 3.22)
project(GPNP_Runtime LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT UNIX OR APPLE)
  message(FATAL_ERROR "This source snapshot uses Linux/POSIX APIs. Build in Linux, WSL, Docker, or RK3588 Linux.")
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d calib3d video videoio highgui imgcodecs)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime installation")
if(NOT ONNXRUNTIME_ROOT)
  foreach(candidate /opt/onnxruntime-linux-x64-gpu-1.20.1 /opt/onnxruntime-linux-aarch64-1.20.1)
    if(EXISTS "${candidate}/include/onnxruntime_cxx_api.h")
      set(ONNXRUNTIME_ROOT "${candidate}")
      break()
    endif()
  endforeach()
endif()
find_library(ORT_LIBRARY NAMES onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "Set ONNXRUNTIME_ROOT to an installation containing include/ and lib/.")
endif()
set(GPNP_WITH_RKNN OFF CACHE BOOL "Build RK3588 NPU backend")
set(RUNTIME_SOURCES
  main.cpp
  src/feature/BinaryCornerExtractor.cpp
  src/feature/TinyTargetExtractor.cpp
  src/matching/MonoCornerIdentity.cpp
  src/matching/StereoFlowMatcher.cpp
  src/pose/InitialPnPSolver.cpp
  src/pose/GPnPSolver.cpp
  src/pose/MonoPnPSolver.cpp
  src/pose/VisualObservation.cpp
  src/pose/GravityNormalEstimator.cpp
  src/pose/GravityConstrainedPnPSolver.cpp
  src/pose/RadarPose.cpp
  src/utils/PoseUtils.cpp
  src/utils/LiveMjpegServer.cpp
  src/tracker/TrackerBase.cpp
  src/tracker/MonoTracker.cpp
  src/tracker/StereoTracker.cpp
  src/tracker/StereoTrackerFlow.cpp
  src/tracker/StereoTrackerDualFlow.cpp
  src/tracker/StereoTrackerDisparity.cpp
  src/tracker/StereoTrackerRobust.cpp
  src/detection/RoiGenerator.cpp
  src/detection/YoloRoiProvider.cpp
  src/detection/RknnDetector.cpp
  src/input/FileStereoSource.cpp
  src/input/DirectoryStereoSource.cpp
  src/input/SequenceSource.cpp
  src/input/CameraSource.cpp
  src/input/VideoSource.cpp
  src/input/FlightImuSource.cpp
  src/input/FlightRadarSource.cpp
  src/input/InputProvider.cpp
  src/input/SimulatedSensors.cpp
  src/fusion/EskfFusionManager.cpp
  src/fusion/VisualPoseFusion.cpp
  src/fusion/ConditionedFlightFusion.cpp
)
add_executable(GPNP ${RUNTIME_SOURCES})
target_include_directories(GPNP PRIVATE include "${CMAKE_CURRENT_SOURCE_DIR}" "${ONNXRUNTIME_ROOT}/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(GPNP PRIVATE Eigen3::Eigen ${OpenCV_LIBS} Threads::Threads "${ORT_LIBRARY}")
set_property(TARGET GPNP PROPERTY BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib")
if(GPNP_WITH_RKNN)
  find_library(RKNN_LIBRARY NAMES rknnrt REQUIRED)
  target_include_directories(GPNP PRIVATE third_party/rknpu2/include)
  target_compile_definitions(GPNP PRIVATE GPNP_RKNN_ENABLED=1)
  target_link_libraries(GPNP PRIVATE "${RKNN_LIBRARY}")
endif()
target_compile_options(GPNP PRIVATE -Wall -Wextra -Wpedantic)
install(TARGETS GPNP DESTINATION bin)
