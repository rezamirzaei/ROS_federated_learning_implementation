cmake_minimum_required(VERSION 3.8)
project(fl_robots_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)

# Generate custom interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  # Messages
  "msg/ModelWeights.msg"
  "msg/TrainingMetrics.msg"
  "msg/RobotStatus.msg"
  "msg/AggregationResult.msg"
  # Services
  "srv/RegisterRobot.srv"
  "srv/TriggerAggregation.srv"
  "srv/GetModelInfo.srv"
  "srv/UpdateHyperparameters.srv"
  # Actions
  "action/TrainRound.action"
  DEPENDENCIES builtin_interfaces std_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

