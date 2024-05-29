cmake_minimum_required(VERSION 3.5.1)

set(MRT_PKG_VERSION 4.0.0)
project(kal2_perception)

###################
## Find packages ##
###################
find_package(mrt_cmake_modules REQUIRED)
include(UseMrtStdCompilerFlags)
include(GatherDeps)

find_package(AutoDeps REQUIRED COMPONENTS ${DEPENDEND_PACKAGES})

mrt_parse_package_xml()

########################
## Add python modules ##
########################
# This adds a python module if located under src/{PROJECT_NAME)
mrt_python_module_setup()


###########
## Build ##
###########
mrt_glob_folders(SRC_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/src")
if (SRC_DIRECTORIES)
    # Found subfolders, add executable for each subfolder
    foreach(SRC_DIR ${SRC_DIRECTORIES})
        mrt_add_executable(${SRC_DIR} FOLDER "src/${SRC_DIR}")
    endforeach()
else()
    # No subfolder found, add executable and python modules for src folder
    mrt_add_executable(${PROJECT_NAME} FOLDER "src")
endif()

#############
## Install ##
#############
# Install all targets, headers by default and scripts and other files if specified (folders or files).
# This command also exports libraries and config files for dependent packages and this supersedes catkin_package.
mrt_install(PROGRAMS scripts FILES res data ${PROJECT_INSTALL_FILES})

#############
## Testing ##
#############
# Add test targets for cpp and python tests
if (CATKIN_ENABLE_TESTING)
    mrt_add_tests(test)
    mrt_add_nosetests(test)
endif()
