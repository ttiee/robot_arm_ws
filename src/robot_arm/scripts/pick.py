#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
import tf.transformations as tf
from moveit_commander import MoveGroupCommander, PlanningSceneInterface
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import Grasp, PlaceLocation
import time
import math

tau = 2 * 3.141592653589793


def open_gripper(posture):
    """打开夹爪"""
    posture.joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
    point = JointTrajectoryPoint()
    point.positions = [0.04, 0.04]
    point.time_from_start = rospy.Duration(0.5)
    posture.points = [point]


def closed_gripper(posture):
    """关闭夹爪"""
    posture.joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
    point = JointTrajectoryPoint()
    point.positions = [0.0, 0.0]
    point.time_from_start = rospy.Duration(0.5)
    posture.points = [point]


def add_collision_objects(scene):
    """添加场景中的碰撞物体"""
    rospy.sleep(1)

    # 地面
    ground_pose = geometry_msgs.msg.PoseStamped()
    ground_pose.header.frame_id = "panda_link0"
    ground_pose.pose.position.x = 0.0
    ground_pose.pose.position.y = 0.0
    ground_pose.pose.position.z = -0.05
    ground_pose.pose.orientation.w = 1.0
    scene.add_box("ground", ground_pose, size=(2.0, 2.0, 0.1))

    # 目标桌子
    table_pose = geometry_msgs.msg.PoseStamped()
    table_pose.header.frame_id = "panda_link0"
    table_pose.pose.position.x = 0.0
    table_pose.pose.position.y = 0.6
    table_pose.pose.position.z = 0.2
    table_pose.pose.orientation.w = 1.0
    scene.add_box("target_table", table_pose, size=(0.6, 0.4, 0.4))

    # 在地面上创建物料（2x2排列，调整位置避免奇异配置）
    objects = []
    for i in range(2):
        for j in range(2):
            object_name = f"object_{i}_{j}"
            object_pose = geometry_msgs.msg.PoseStamped()
            object_pose.header.frame_id = "panda_link0"
            # 优化位置，避免奇异配置
            object_pose.pose.position.x = 0.35 + i * 0.1  # 增大间距，避免过于接近
            object_pose.pose.position.y = -0.15 + j * 0.1  # 调整Y位置范围
            object_pose.pose.position.z = 0.025  # 物体高度的一半
            object_pose.pose.orientation.w = 1.0
            scene.add_box(object_name, object_pose, size=(0.04, 0.04, 0.05))
            objects.append(object_name)

    rospy.sleep(2)
    return objects


def create_multiple_grasps(x, y, z, object_name):
    """创建多个抓取姿态以提高成功率"""
    grasps = []

    # 基本的垂直抓取姿态
    for angle_offset in [0, math.pi / 6, -math.pi / 6, math.pi / 4, -math.pi / 4]:
        grasp = Grasp()
        grasp.id = f"grasp_{object_name}_{len(grasps)}"

        # 设置抓取姿态 - 垂直向下，但增加一些角度变化
        grasp.grasp_pose.header.frame_id = "panda_link0"
        quat = tf.quaternion_from_euler(-tau / 2, 0, angle_offset)
        grasp.grasp_pose.pose.orientation.x = quat[0]
        grasp.grasp_pose.pose.orientation.y = quat[1]
        grasp.grasp_pose.pose.orientation.z = quat[2]
        grasp.grasp_pose.pose.orientation.w = quat[3]

        # 抓取位置，增加一些位置变化
        x_offset = 0.005 * math.cos(angle_offset)
        y_offset = 0.005 * math.sin(angle_offset)
        grasp.grasp_pose.pose.position.x = x + x_offset
        grasp.grasp_pose.pose.position.y = y + y_offset
        grasp.grasp_pose.pose.position.z = z + 0.06  # 稍微提高抓取高度

        # 预抓取接近
        grasp.pre_grasp_approach.direction.header.frame_id = "panda_link0"
        grasp.pre_grasp_approach.direction.vector.z = -1.0
        grasp.pre_grasp_approach.min_distance = 0.06
        grasp.pre_grasp_approach.desired_distance = 0.1

        # 抓取后撤退
        grasp.post_grasp_retreat.direction.header.frame_id = "panda_link0"
        grasp.post_grasp_retreat.direction.vector.z = 1.0
        grasp.post_grasp_retreat.min_distance = 0.1
        grasp.post_grasp_retreat.desired_distance = 0.15

        # 夹爪姿态
        open_gripper(grasp.pre_grasp_posture)
        closed_gripper(grasp.grasp_posture)

        # 设置质量
        grasp.grasp_quality = 1.0 - 0.1 * len(grasps)  # 优先级递减

        grasps.append(grasp)

    # 添加侧面抓取姿态
    for side_angle in [math.pi / 3, 2 * math.pi / 3, 4 * math.pi / 3, 5 * math.pi / 3]:
        grasp = Grasp()
        grasp.id = f"grasp_{object_name}_{len(grasps)}_side"

        # 侧面抓取姿态
        grasp.grasp_pose.header.frame_id = "panda_link0"
        quat = tf.quaternion_from_euler(-math.pi / 3, 0, side_angle)
        grasp.grasp_pose.pose.orientation.x = quat[0]
        grasp.grasp_pose.pose.orientation.y = quat[1]
        grasp.grasp_pose.pose.orientation.z = quat[2]
        grasp.grasp_pose.pose.orientation.w = quat[3]

        # 侧面接近位置
        approach_dist = 0.08
        grasp.grasp_pose.pose.position.x = x + approach_dist * math.cos(side_angle)
        grasp.grasp_pose.pose.position.y = y + approach_dist * math.sin(side_angle)
        grasp.grasp_pose.pose.position.z = z + 0.03

        # 预抓取接近 - 水平接近
        grasp.pre_grasp_approach.direction.header.frame_id = "panda_link0"
        grasp.pre_grasp_approach.direction.vector.x = -math.cos(side_angle)
        grasp.pre_grasp_approach.direction.vector.y = -math.sin(side_angle)
        grasp.pre_grasp_approach.direction.vector.z = 0.0
        grasp.pre_grasp_approach.min_distance = 0.04
        grasp.pre_grasp_approach.desired_distance = 0.08

        # 抓取后撤退 - 向上撤退
        grasp.post_grasp_retreat.direction.header.frame_id = "panda_link0"
        grasp.post_grasp_retreat.direction.vector.x = 0.0
        grasp.post_grasp_retreat.direction.vector.y = 0.0
        grasp.post_grasp_retreat.direction.vector.z = 1.0
        grasp.post_grasp_retreat.min_distance = 0.1
        grasp.post_grasp_retreat.desired_distance = 0.15

        # 夹爪姿态
        open_gripper(grasp.pre_grasp_posture)
        closed_gripper(grasp.grasp_posture)

        grasp.grasp_quality = 0.5  # 侧面抓取优先级较低

        grasps.append(grasp)

    return grasps


def create_place_location(x, y, z):
    """创建放置位置"""
    location = PlaceLocation()

    # 设置放置姿态
    location.place_pose.header.frame_id = "panda_link0"
    quat = tf.quaternion_from_euler(0, 0, 0)
    location.place_pose.pose.orientation.x = quat[0]
    location.place_pose.pose.orientation.y = quat[1]
    location.place_pose.pose.orientation.z = quat[2]
    location.place_pose.pose.orientation.w = quat[3]
    location.place_pose.pose.position.x = x
    location.place_pose.pose.position.y = y
    location.place_pose.pose.position.z = z

    # 预放置接近
    location.pre_place_approach.direction.header.frame_id = "panda_link0"
    location.pre_place_approach.direction.vector.z = -1.0
    location.pre_place_approach.min_distance = 0.08
    location.pre_place_approach.desired_distance = 0.12

    # 放置后撤退
    location.post_place_retreat.direction.header.frame_id = "panda_link0"
    location.post_place_retreat.direction.vector.z = 1.0
    location.post_place_retreat.min_distance = 0.1
    location.post_place_retreat.desired_distance = 0.15

    # 夹爪姿态
    open_gripper(location.post_place_posture)

    return location


def pick_object_with_fallback(group, object_name, x, y, z):
    """带回退策略的抓取函数"""
    print(f"正在抓取 {object_name}...")

    # 尝试多种抓取姿态
    grasps = create_multiple_grasps(x, y, z, object_name)

    group.set_support_surface_name("ground")
    result = group.pick(object_name, grasps)

    if result == 1:
        print(f"成功抓取 {object_name}")
        return True
    else:
        print(f"使用标准抓取失败，尝试移动到中间位置后再抓取...")

        # 回退策略：先移动到安全位置
        group.set_named_target("ready")
        group.go(wait=True)
        rospy.sleep(1)

        # 尝试更简单的抓取策略
        simple_grasps = create_multiple_grasps(x, y, z + 0.02, object_name)[
            :3
        ]  # 只使用前3个最好的抓取姿态
        result = group.pick(object_name, simple_grasps)

        if result == 1:
            print(f"回退策略成功抓取 {object_name}")
            return True
        else:
            print(f"抓取 {object_name} 最终失败，错误代码: {result}")
            return False


def place_object(group, object_name, x, y, z):
    """放置单个物体"""
    print(f"正在放置 {object_name}...")

    place_locations = []
    location = create_place_location(x, y, z)
    place_locations.append(location)

    group.set_support_surface_name("target_table")
    result = group.place(object_name, place_locations)

    if result == 1:
        print(f"成功放置 {object_name}")
        return True
    else:
        print(f"放置 {object_name} 失败，错误代码: {result}")
        return False


def move_objects_to_table(group, objects):
    """将所有物体从地面搬运到桌子上"""
    successful_moves = 0

    # 定义桌子上的放置位置（2x2网格，增大间距）
    table_positions = []
    for i in range(2):
        for j in range(2):
            x = -0.08 + i * 0.12  # 增大X方向间距
            y = 0.48 + j * 0.12  # 增大Y方向间距
            z = 0.425  # 桌子表面高度 + 物体高度的一半
            table_positions.append((x, y, z))

    # 地面物体的位置（与add_collision_objects中的位置对应）
    ground_positions = []
    for i in range(2):
        for j in range(2):
            x = 0.35 + i * 0.1
            y = -0.15 + j * 0.1
            z = 0.025
            ground_positions.append((x, y, z))

    # 先移动到准备位置
    print("移动到准备位置...")
    group.set_named_target("ready")
    group.go(wait=True)
    rospy.sleep(2)

    # 逐个搬运物体
    for idx, object_name in enumerate(objects):
        if idx >= len(table_positions):
            print(f"桌子空间不足，跳过 {object_name}")
            continue

        ground_pos = ground_positions[idx]
        table_pos = table_positions[idx]

        print(f"处理第 {idx+1}/{len(objects)} 个物体: {object_name}")

        # 抓取物体
        if pick_object_with_fallback(
            group, object_name, ground_pos[0], ground_pos[1], ground_pos[2]
        ):
            rospy.sleep(2)  # 等待抓取稳定

            # 放置物体
            if place_object(
                group, object_name, table_pos[0], table_pos[1], table_pos[2]
            ):
                successful_moves += 1
                rospy.sleep(2)  # 等待放置稳定
            else:
                print(f"放置失败，物体 {object_name} 可能仍在夹爪中")

        # 操作间隔，让系统稳定
        rospy.sleep(3)

    print(f"搬运完成！成功搬运了 {successful_moves}/{len(objects)} 个物体")
    return successful_moves


def main():
    """主函数"""
    print("初始化Panda机器人多物体搬运仿真...")

    # 初始化MoveIt
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("panda_multi_pick_place", anonymous=True)

    # 创建场景和运动规划组
    scene = PlanningSceneInterface()
    group = MoveGroupCommander("panda_arm")

    # 优化规划参数
    group.set_planning_time(60.0)  # 增加规划时间
    group.set_num_planning_attempts(15)  # 增加规划尝试次数
    group.set_max_velocity_scaling_factor(0.3)  # 降低速度，提高精度
    group.set_max_acceleration_scaling_factor(0.3)  # 降低加速度
    group.set_goal_position_tolerance(0.01)  # 设置位置容差
    group.set_goal_orientation_tolerance(0.05)  # 设置姿态容差

    # 设置规划器
    group.set_planner_id("RRTConnect")

    print("设置场景...")
    rospy.sleep(2)

    # 清除之前的场景
    scene.clear()
    rospy.sleep(1)

    # 添加碰撞物体
    objects = add_collision_objects(scene)

    print("开始搬运作业...")
    rospy.sleep(3)  # 确保场景完全加载

    # 执行搬运任务
    successful_count = move_objects_to_table(group, objects)

    # 最后移动到home位置
    print("返回初始位置...")
    group.set_named_target("ready")
    group.go(wait=True)

    print(f"所有搬运任务完成！最终成功率: {successful_count}/{len(objects)}")

    # 保持节点运行以便观察结果
    rospy.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        print("程序被中断")
    except Exception as e:
        print(f"程序出现错误: {e}")
