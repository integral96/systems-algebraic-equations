cmake_minimum_required(VERSION 3.5)

project(system_solver LANGUAGES CXX)

set(SOURCES
    main.cpp)
set(HEADERS
    solver_algebrick.hpp
    meta_solver_algebrick.hpp
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth=14800")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

find_package(Boost ${BOOST_MIN_VERSION} COMPONENTS context system regex)

target_link_libraries (${PROJECT_NAME} ${Boost_LIBRARIES})
