### Panda 机械臂抓取放置仿真项目 (基础框架)

这是一个使用 ROS 和 MoveIt\! 进行 Panda 机械臂抓取放置任务的基础框架。项目的主要目标是演示如何搭建一个仿真环境，并使用 Python 脚本通过 MoveIt\! 来规划机械臂的运动。

**重要提示**: 目前，抓取和放置的动作仅在 RViz 的 **规划层面** 得以实现。Gazebo 中的物理夹爪尚 **不能** 实际抓住或放下物体。因此，本项目主要用于学习和演示 MoveIt\! 的基本API调用和运动规划流程，而不是一个功能完整的物理仿真。

### 当前实现的功能

  * [cite\_start]启动一个包含 Panda 机械臂和几个小物块的 Gazebo 仿真世界。 [cite: 1, 11]
  * [cite\_start]加载 MoveIt\!，用于机械臂的运动规划，并通过 RViz 进行可视化。 [cite: 1, 11]
  * 提供一个 Python 脚本 (`pick.py`)，该脚本定义了连接 MoveIt\!、添加场景物体、以及规划“抓取”和“放置”动作的逻辑。
  * 运行脚本后，可以在 RViz 中观察到机械臂模型按照预定逻辑执行了抓取和放置的 **规划动画**。

### 复现步骤

#### 1\. 环境依赖

请确保已安装 ROS Noetic 及以下软件包：

```bash
# 安装 MoveIt!
sudo apt update
sudo apt install ros-noetic-moveit

# 安装 Panda 机器人相关包
sudo apt install ros-noetic-franka-ros ros-noetic-panda-moveit-config

# 安装 catkin 构建工具
sudo apt install ros-noetic-catkin python3-catkin-tools
```

#### 2\. 编译项目

1.  **创建并进入Catkin工作区**:

    ```bash
    mkdir -p ~/catkin_ws/src
    cd ~/catkin_ws/src
    ```

2.  **放入项目文件**:
    将 `robot_arm` 文件夹复制到当前 `src` 目录下。

3.  **安装依赖并编译**:

    ```bash
    cd ~/catkin_ws
    rosdep install --from-paths src --ignore-src -r -y
    catkin_make
    ```

4.  **Source环境变量**:

    ```bash
    source ~/catkin_ws/devel/setup.bash
    ```

    建议将此命令添加到 `~/.bashrc` 中，以免每次都要手动执行。

#### 3\. 运行与观察

整个流程需要两个终端。

1.  **终端 A：启动仿真和 MoveIt\!**
    运行 `spawn_blocks.launch` 文件来启动 Gazebo 环境、加载机器人和 MoveIt\!。

    ```bash
    roslaunch robot_arm spawn_blocks.launch
    ```

    执行后，Gazebo 和 RViz 窗口会自动打开。

2.  **终端 B：运行规划脚本**
    等待上一步的 RViz 和 Gazebo 完全加载后，在 **新的终端** 中运行 `pick.py`。

    ```bash
    rosrun robot_arm pick.py
    ```

### 预期结果

  * **在 Gazebo 中**：你会看到 Panda 机械臂、地面和四个方块。当 `pick.py` 运行时，机械臂会移动到方块上方，再移动到桌子上方。但是，**夹爪不会有开合动作，方块也不会被拿起**。
  * **在 RViz 中**：你会看到机械臂的规划模型。当 `pick.py` 运行时，你可以观察到虚拟的机械臂模型完整地模拟了接近物体、闭合夹爪（模型颜色变化）、拾取、移动、放置和张开夹爪的全过程动画。

这清晰地表明了项目的当前状态：**规划成功，但物理执行（特别是夹爪部分）尚未实现**。

### 文件结构简介

```
robot_arm/
├── launch/
│   ├── spawn_blocks.launch   # 主启动文件，负责启动所有节点
│   └── panda_gazebo.launch   # 启动多个panda的示例文件
├── scripts/
│   └── pick.py               # 核心规划逻辑脚本
├── urdf/
│   └── block.urdf.xacro      # 方块的模型文件
├── CMakeLists.txt
└── package.xml
```

### 后续工作建议

要让这个项目在 Gazebo 中实现真正的物理抓取，可以从以下几个方面入手：

1.  **配置夹爪控制器**：需要为 Panda 的夹爪配置 `gazebo_ros_control`，并确保夹爪的 effort/position controller 能够正确加载。
2.  **连接脚本与控制器**：修改 `pick.py` 中 `open_gripper` 和 `closed_gripper` 函数，使其不再是生成轨迹消息，而是向夹爪的控制器发布实际的控制指令（例如 `std_msgs/Float64MultiArray` 或 `control_msgs/GripperCommand`）。
3.  **Gazebo 抓取插件**：一种更可靠的方法是使用或编写一个 Gazebo 插件，该插件可以在夹爪闭合并接触到物体时，动态地创建一个 "fixed" 关节，将物体“吸附”到夹爪上，在放置时再解除该关节。