#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from copy import deepcopy

tau = 2 * math.pi


def open_gripper(posture):
    posture.joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
    point = JointTrajectoryPoint()
    point.positions = [0.04, 0.04]
    point.time_from_start = rospy.Duration(0.5)
    posture.points = [point]


def closed_gripper(posture):
    posture.joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
    point = JointTrajectoryPoint()
    point.positions = [0.0, 0.0]
    point.time_from_start = rospy.Duration(0.5)
    posture.points = [point]


def add_collision_objects(scene, objects_info):
    rospy.sleep(1)
    ground_pose = geometry_msgs.msg.PoseStamped()
    ground_pose.header.frame_id = "panda_link0"
    ground_pose.pose.position.z = -0.05
    ground_pose.pose.orientation.w = 1.0
    scene.add_box("ground", ground_pose, size=(2.0, 2.0, 0.1))

    table_pose = geometry_msgs.msg.PoseStamped()
    table_pose.header.frame_id = "panda_link0"
    table_pose.pose.position.x = 0.0
    table_pose.pose.position.y = 0.6
    table_pose.pose.position.z = 0.2
    table_pose.pose.orientation.w = 1.0
    scene.add_box("target_table", table_pose, size=(0.6, 0.4, 0.4))

    for name, info in objects_info.items():
        object_pose = geometry_msgs.msg.PoseStamped()
        object_pose.header.frame_id = "panda_link0"
        pos = info['ground_pos']
        object_pose.pose.position.x = pos[0]
        object_pose.pose.position.y = pos[1]
        object_pose.pose.position.z = pos[2]
        object_pose.pose.orientation.w = 1.0
        scene.add_box(name, object_pose, size=(0.04, 0.04, 0.05))

    rospy.sleep(2)
    return list(objects_info.keys())


def create_multiple_grasps(x, y, z, object_name):
    grasps = []
    quat_base = tf.quaternion_from_euler(-math.pi, 0, 0)

    for angle in [0, math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2]:
        grasp = Grasp()
        grasp.id = f"grasp_{object_name}_{len(grasps)}"
        grasp.grasp_pose.header.frame_id = "panda_link0"
        quat = tf.quaternion_multiply(quat_base, tf.quaternion_from_euler(0, 0, angle))
        grasp.grasp_pose.pose.orientation = geometry_msgs.msg.Quaternion(*quat)
        
        grasp.grasp_pose.pose.position.x = x
        grasp.grasp_pose.pose.position.y = y
        grasp.grasp_pose.pose.position.z = z + 0.1
        
        # *** 增加接近和撤退的距离，给予规划器更多空间 ***
        grasp.pre_grasp_approach.direction.header.frame_id = "panda_link0"
        grasp.pre_grasp_approach.direction.vector.z = -1.0
        grasp.pre_grasp_approach.min_distance = 0.1
        grasp.pre_grasp_approach.desired_distance = 0.15

        grasp.post_grasp_retreat.direction.header.frame_id = "panda_link0"
        grasp.post_grasp_retreat.direction.vector.z = 1.0
        grasp.post_grasp_retreat.min_distance = 0.15
        grasp.post_grasp_retreat.desired_distance = 0.25

        open_gripper(grasp.pre_grasp_posture)
        closed_gripper(grasp.grasp_posture)
        grasps.append(grasp)

    return grasps


def create_multiple_place_locations(x, y, z):
    locations = []
    for angle in [0, math.pi / 4, -math.pi / 4]:
        location = PlaceLocation()
        location.id = f"place_{len(locations)}"
        location.place_pose.header.frame_id = "panda_link0"
        quat = tf.quaternion_from_euler(0, 0, angle)
        location.place_pose.pose.orientation = geometry_msgs.msg.Quaternion(*quat)
        location.place_pose.pose.position.x = x
        location.place_pose.pose.position.y = y
        location.place_pose.pose.position.z = z
        location.pre_place_approach.direction.header.frame_id = "panda_link0"
        location.pre_place_approach.direction.vector.z = -1.0
        location.pre_place_approach.min_distance = 0.08
        location.pre_place_approach.desired_distance = 0.12
        location.post_place_retreat.direction.header.frame_id = "panda_link0"
        location.post_place_retreat.direction.vector.z = 1.0
        location.post_place_retreat.min_distance = 0.1
        location.post_place_retreat.desired_distance = 0.2
        open_gripper(location.post_place_posture)
        locations.append(location)
    return locations


def pick_object(group, scene, object_name, x, y, z):
    print(f"正在尝试抓取 {object_name}...")
    grasps = create_multiple_grasps(x, y, z, object_name)
    
    for _ in range(5):
        result = group.pick(object_name, grasps)
        if result == 1:
            print(f"成功抓取 {object_name}")
            return True
        print(f"抓取失败，错误代码: {result}。正在重试...")
        rospy.sleep(1)
        
    print(f"多次抓取失败，执行回退策略：移动到 'ready' 位置再试。")
    group.set_named_target("ready")
    group.go(wait=True)
    rospy.sleep(1)
    
    result = group.pick(object_name, grasps)
    if result == 1:
        print(f"回退策略成功抓取 {object_name}")
        return True
    else:
        print(f"抓取 {object_name} 最终失败，错误代码: {result}")
        scene.remove_world_object(object_name)
        return False


def place_object_with_fallback(group, scene, object_name, x, y, z):
    print(f"正在尝试放置 {object_name}...")
    place_locations = create_multiple_place_locations(x, y, z)
    
    for _ in range(3):
        result = group.place(object_name, place_locations)
        if result == 1:
            print(f"成功放置 {object_name}")
            return True
        print(f"放置失败，错误代码: {result}。重试中...")
        rospy.sleep(1)

    print(f"多次放置失败，执行回退策略...")
    group.set_named_target("ready")
    group.go(wait=True)
    
    print("在安全位置释放物体...")
    scene.remove_attached_object(group.get_end_effector_link(), name=object_name)
    scene.remove_world_object(object_name)
    print(f"放置 {object_name} 最终失败。已从场景中移除该物体。")
    return False


def move_objects_to_table(group, scene, objects_info):
    successful_moves = 0
    sorted_objects = sorted(
        objects_info.keys(),
        key=lambda name: math.sqrt(objects_info[name]['ground_pos'][0]**2 + objects_info[name]['ground_pos'][1]**2),
        reverse=True
    )

    for object_name in sorted_objects:
        info = objects_info[object_name]
        ground_pos = info['ground_pos']
        table_pos = info['table_pos']
        
        print("\n" + "="*30)
        print(f"开始处理物体: {object_name}")
        print("="*30)
        
        print("移动到 'ready' 准备姿态...")
        group.set_named_target("ready")
        group.go(wait=True)
        rospy.sleep(1)

        if pick_object(group, scene, object_name, ground_pos[0], ground_pos[1], ground_pos[2]):
            rospy.sleep(1)
            print("抓取成功，移动到中间航点...")
            group.set_named_target("ready")
            group.go(wait=True)
            rospy.sleep(1)
            if place_object_with_fallback(group, scene, object_name, table_pos[0], table_pos[1], table_pos[2]):
                successful_moves += 1
                rospy.sleep(1)
            else:
                print(f"未能放置 {object_name}，继续下一个物体。")
        else:
            print(f"未能抓取 {object_name}，跳过该物体。")
        rospy.sleep(2)

    print("\n" + "="*30)
    print(f"搬运完成！成功搬运了 {successful_moves}/{len(objects_info)} 个物体")
    print("="*30)
    return successful_moves


def main():
    print("初始化Panda机器人多物体搬运仿真...")
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("panda_multi_pick_place", anonymous=True)
    scene = PlanningSceneInterface()
    group = MoveGroupCommander("panda_arm")
    
    # *** 优化建议 2 & 3: 增加规划时间并更换为更强大的规划器 ***
    group.set_planning_time(30.0) 
    group.set_planner_id("RRTstar") # 使用RRTstar
    
    group.set_num_planning_attempts(10)
    group.set_max_velocity_scaling_factor(0.5)
    group.set_max_acceleration_scaling_factor(0.5)
    group.set_goal_position_tolerance(0.005)
    group.set_goal_orientation_tolerance(0.01)

    # *** 优化建议 1: 将物体放置在机械臂更舒适的工作空间内 ***
    objects_info = {
        "object_0_0": {"ground_pos": [0.45, -0.15, 0.025], "table_pos": [-0.08, 0.48, 0.425]},
        "object_0_1": {"ground_pos": [0.45, -0.05, 0.025], "table_pos": [-0.08, 0.60, 0.425]},
        "object_1_0": {"ground_pos": [0.55, -0.15, 0.025], "table_pos": [0.04, 0.48, 0.425]},
        "object_1_1": {"ground_pos": [0.55, -0.05, 0.025], "table_pos": [0.04, 0.60, 0.425]},
    }

    print("清理并设置场景...")
    scene.clear()
    rospy.sleep(1)
    
    attached_objects = scene.get_attached_objects()
    for obj in attached_objects.keys():
        scene.remove_attached_object(group.get_end_effector_link(), name=obj)
        rospy.sleep(0.5)

    add_collision_objects(scene, objects_info)
    
    print("开始搬运作业...")
    rospy.sleep(2)

    successful_count = move_objects_to_table(group, scene, objects_info)

    print("所有任务完成，返回初始位置...")
    group.set_named_target("ready")
    group.go(wait=True)
    
    print(f"最终成功率: {successful_count}/{len(objects_info)}")
    rospy.sleep(5)
    
    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        print("程序被中断")
    except Exception as e:
        import traceback
        print(f"程序出现未处理的错误: {e}")
        traceback.print_exc()