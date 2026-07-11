FROM ros:jazzy-ros-base-noble

ENV ROS_DISTRO=jazzy

# Nav stack + TB3 sim + SLAM + viz bridge + Gazebo Harmonic (via ros_gz).
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-navigation2 \
      ros-jazzy-nav2-bringup \
      ros-jazzy-slam-toolbox \
      ros-jazzy-turtlebot3-gazebo \
      ros-jazzy-turtlebot3-navigation2 \
      ros-jazzy-foxglove-bridge \
      ros-jazzy-ros-gz \
      python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts

# rosdep before build, every time (ros2 skill key directive).
RUN . /opt/ros/${ROS_DISTRO}/setup.sh \
 && apt-get update \
 && rosdep update --rosdistro ${ROS_DISTRO} \
 && rosdep install --from-paths src -y --ignore-src \
 && colcon build --symlink-install \
 && rm -rf /var/lib/apt/lists/*

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh /ws/scripts/*.sh 2>/dev/null || chmod +x /entrypoint.sh

ENV TURTLEBOT3_MODEL=burger
EXPOSE 8765
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
