cmake_minimum_required(VERSION 3.20)
project(mera_core LANGUAGES CXX)
cmake_policy(SET CMP0167 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenSSL REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(Boost REQUIRED)

add_executable(mera_core mera_core.cpp)
target_link_libraries(mera_core PRIVATE OpenSSL::Crypto SQLite::SQLite3 Boost::boost)

if (MSVC)
  target_compile_options(mera_core PRIVATE /W4 /EHsc)
else()
  target_compile_options(mera_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
