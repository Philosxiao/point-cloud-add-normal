cmake_minimum_required(VERSION 2.8.10)

SET(CMAKE_BUILD_TYPE "Debug") 

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")

SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")



project(check_bar LANGUAGES CXX)

find_package(OpenCV 3.4.0 REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

add_executable(run_test main.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
 
# find_package(Boost REQUIRED COMPONENTS system thread filesystem)
# include_directories(${Boost_INCLUDE_DIRS})
# target_link_libraries(run_test ${Boost_LIBRARIES})

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(run_test ${OpenCV_LIBRARIES})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_libraries(run_test ${PCL_LIBRARIES})
target_link_libraries(run_test ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_FILTERS_LIBRARIES})