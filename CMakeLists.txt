CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(face-direction-detector)

FIND_PACKAGE( OpenCV REQUIRED )
FIND_PACKAGE( flandmark REQUIRED )

# Define flags for specific OS building
IF(UNIX)
  ADD_DEFINITIONS(-DUNIX)
  IF(APPLE)
    ADD_DEFINITIONS(-DAPPLE)
    MESSAGE("Building on APPLE")
    SET(JSONCPP_INCLUDE_DIR /Users/aloschil/workspace/jsoncpp-src-0.5.0/include)    
    SET(JSONCPP_LIB_DIR /Users/aloschil/workspace/jsoncpp-src-0.5.0/libs/linux-gcc-4.2.1)
    SET(JSONCPP_LIBRARIES libjson_linux-gcc-4.2.1_libmt.dylib)
  ELSE(APPLE)
    SET(LINUX 1)
    ADD_DEFINITIONS(-DLINUX)
    MESSAGE("Building on LINUX")
    SET(JSONCPP_LIBRARIES jsoncpp)
  ENDIF(APPLE)
ENDIF(UNIX)

IF(WIN32)
  ADD_DEFINITIONS(-DWIN32)
  MESSAGE("Building on WIN32")
ENDIF(WIN32)

IF(UNIX)
  #add definitions, compiler switches, etc.
  ADD_DEFINITIONS(-Wall -Wno-deprecated)
  #SET(CMAKE_CXX_FLAGS_DEBUG "-g")
  SET(CMAKE_SKIP_RPATH ON)

  IF(CMAKE_COMPILER_IS_GNUCC)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fmessage-length=0")
  ENDIF(CMAKE_COMPILER_IS_GNUCC)
  IF(CMAKE_COMPILER_IS_GNUCXX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmessage-length=0")
  ENDIF(CMAKE_COMPILER_IS_GNUCXX)
ENDIF(UNIX)

# If you don't want the full compiler output, remove the following line
SET(CMAKE_VERBOSE_MAKEFILE ON)
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS "-D_ENGLISH")

######## INCLUDE AND LINK DIRECTORIES ########

INCLUDE_DIRECTORIES(
  /usr/local/include
  ${JSONCPP_INCLUDE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OPENCV_INCLUDE_DIR}
  ${FLANDMARK_INCLUDE_DIRS}
)

LINK_DIRECTORIES(
  ${FLANDMARK_LINK_DIRS}
  ${OpenCV_LIB_DIR}
  ${JSONCPP_LIB_DIR}
)


######## MAIN EXECUTABLES ########

add_executable(face-direction-detector main.cxx)
TARGET_LINK_LIBRARIES(face-direction-detector ${AMT_LIBS} ${FLANDMARK_LIBRARIES} ${OpenCV_LIBS} ${JSONCPP_LIBRARIES})
