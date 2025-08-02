#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
import tf.transformations as tf
from moveit_commander import MoveGroupCommander, PlanningSceneInterface
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import Grasp, PlaceLocation, RobotTrajectory
import time
import math
import pickle
import os
from copy import deepcopy
import hashlib
import rospkg

tau = 2 * math.pi

class TrajectoryCache:
    """轨迹缓存管理类"""
    def __init__(self, cache_dir=None):
        if cache_dir is None:
            rospack = rospkg.RosPack()
            try:
                package_path = rospack.get_path('robot_arm')
                cache_dir = os.path.join(package_path, 'trajectory_cache')
            except rospkg.ResourceNotFound:
                cache_dir = "trajectory_cache"
        
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        print(f"轨迹缓存目录: {self.cache_dir}")
    
    def _get_cache_key(self, start_pose, end_pose, grasp_type="pick"):
        """根据起始和结束位置生成缓存键"""
        pose_str = f"{grasp_type}_{start_pose}_{end_pose}"
        return hashlib.md5(pose_str.encode()).hexdigest()
    
    def save_trajectory(self, trajectory, start_pose, end_pose, grasp_type="pick"):
        """保存轨迹到文件"""
        cache_key = self._get_cache_key(start_pose, end_pose, grasp_type)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(trajectory, f)
            print(f"轨迹已保存: {cache_key}")
            return True
        except Exception as e:
            print(f"保存轨迹失败: {e}")
            return False
    
    def load_trajectory(self, start_pose, end_pose, grasp_type="pick"):
        """从文件加载轨迹"""
        cache_key = self._get_cache_key(start_pose, end_pose, grasp_type)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                trajectory = pickle.load(f)
            print(f"轨迹已加载: {cache_key}")
            return trajectory
        except Exception as e:
            print(f"加载轨迹失败: {e}")
            return None
    
    def save_named_target_trajectory(self, trajectory, target_name):
        """保存命名目标的轨迹"""
        cache_file = os.path.join(self.cache_dir, f"named_target_{target_name}.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(trajectory, f)
            print(f"命名目标轨迹已保存: {target_name}")
            return True
        except Exception as e:
            print(f"保存命名目标轨迹失败: {e}")
            return False
    
    def load_named_target_trajectory(self, target_name):
        """加载命名目标的轨迹"""
        cache_file = os.path.join(self.cache_dir, f"named_target_{target_name}.pkl")
        
        if not os.path.exists(cache_file):
            print(f"未找到缓存文件: {cache_file}")
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                trajectory = pickle.load(f)
            print(f"命名目标轨迹已加载: {target_name}")
            return trajectory
        except Exception as e:
            print(f"加载命名目标轨迹失败: {e}")
            return None

    def clear_cache(self):
        """清空缓存"""
        for file in os.listdir(self.cache_dir):
            if file.endswith('.pkl'):
                os.remove(os.path.join(self.cache_dir, file))
        print("轨迹缓存已清空")


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


def pose_to_string(pose):
    """将pose转换为字符串用于缓存"""
    if isinstance(pose, (list, tuple)):
        return f"[{pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}]"
    else:
        return f"[{pose.position.x:.3f},{pose.position.y:.3f},{pose.position.z:.3f}]"


def execute_trajectory(group, trajectory):
    """执行轨迹"""
    try:
        success = group.execute(trajectory, wait=True)
        group.stop()
        return success
    except Exception as e:
        print(f"执行轨迹失败: {e}")
        return False


def plan_trajectory(group, target_pose):
    """统一的轨迹规划函数"""
    group.set_pose_target(target_pose)
    
    try:
        plan_result = group.plan()
        
        if isinstance(plan_result, tuple):
            success, trajectory, planning_time, error_code = plan_result
            if success and trajectory:
                print(f"规划成功，用时: {planning_time:.2f}s")
                return trajectory
        elif hasattr(plan_result, 'joint_trajectory') and len(plan_result.joint_trajectory.points) > 0:
            print(f"规划成功")
            return plan_result
        elif isinstance(plan_result, list) and len(plan_result) > 0:
            print(f"规划成功")
            return plan_result
            
        print(f"规划失败")
        return None
        
    except Exception as e:
        print(f"规划过程出错: {e}")
        return None


def plan_with_cache(group, cache, target_pos, trajectory_type="move"):
    """通用的带缓存轨迹规划函数"""
    current_pose = group.get_current_pose().pose
    current_pos_str = pose_to_string(current_pose)
    target_pos_str = pose_to_string(target_pos)
    
    # 尝试加载缓存
    cached_trajectory = cache.load_trajectory(current_pos_str, target_pos_str, trajectory_type)
    if cached_trajectory:
        print(f"使用缓存的{trajectory_type}轨迹")
        return cached_trajectory
    
    # 规划新轨迹
    print(f"规划新的{trajectory_type}轨迹")
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = target_pos[0]
    target_pose.position.y = target_pos[1]
    target_pose.position.z = target_pos[2]
    target_pose.orientation.x = 1.0
    target_pose.orientation.y = 0.0
    target_pose.orientation.z = 0.0
    target_pose.orientation.w = 0.0
    
    trajectory = plan_trajectory(group, target_pose)
    if trajectory:
        cache.save_trajectory(trajectory, current_pos_str, target_pos_str, trajectory_type)
    
    return trajectory


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


def move_to_ready_with_cache(group, cache):
    """使用缓存移动到ready位置"""
    cached_trajectory = cache.load_named_target_trajectory("ready")
    if cached_trajectory:
        print("使用缓存的ready轨迹")
        if execute_trajectory(group, cached_trajectory):
            return True
    
    print("规划新的ready轨迹")
    group.set_named_target("ready")
    
    try:
        plan_result = group.plan()
        
        if isinstance(plan_result, tuple):
            success, trajectory, planning_time, error_code = plan_result
            if success and trajectory:
                print(f"ready轨迹规划成功，用时: {planning_time:.2f}s")
                if group.execute(trajectory, wait=True):
                    group.stop()
                    cache.save_named_target_trajectory(trajectory, "ready")
                    return True
        elif hasattr(plan_result, 'joint_trajectory') and len(plan_result.joint_trajectory.points) > 0:
            if group.execute(plan_result, wait=True):
                group.stop()
                cache.save_named_target_trajectory(plan_result, "ready")
                return True
    except Exception as e:
        print(f"ready轨迹规划/执行失败: {e}")
    
    return group.go(wait=True)


def pick_object_with_cache(group, scene, cache, object_name, x, y, z):
    """使用缓存的抓取功能"""
    print(f"正在尝试抓取 {object_name}...")
    
    target_pos = [x, y, z + 0.15]
    trajectory = plan_with_cache(group, cache, target_pos, "pick")
    
    if trajectory and execute_trajectory(group, trajectory):
        print(f"成功移动到抓取位置")
        grasps = create_multiple_grasps(x, y, z, object_name)
        result = group.pick(object_name, grasps)
        if result == 1:
            print(f"成功抓取 {object_name}")
            return True
    
    print(f"缓存轨迹失败，使用原始抓取方法")
    grasps = create_multiple_grasps(x, y, z, object_name)
    
    for _ in range(3):
        result = group.pick(object_name, grasps)
        if result == 1:
            print(f"成功抓取 {object_name}")
            return True
        print(f"抓取失败，错误代码: {result}。正在重试...")
        rospy.sleep(0.5)
        
    print(f"抓取 {object_name} 失败")
    scene.remove_world_object(object_name)
    return False


def place_object_with_cache(group, scene, cache, object_name, x, y, z):
    """使用缓存的放置功能"""
    print(f"正在尝试放置 {object_name}...")
    
    target_pos = [x, y, z + 0.1]
    trajectory = plan_with_cache(group, cache, target_pos, "place")
    
    if trajectory and execute_trajectory(group, trajectory):
        print(f"成功移动到放置位置")
        place_locations = create_multiple_place_locations(x, y, z)
        result = group.place(object_name, place_locations)
        if result == 1:
            print(f"成功放置 {object_name}")
            return True
    
    print(f"缓存轨迹失败，使用原始放置方法")
    place_locations = create_multiple_place_locations(x, y, z)
    
    for _ in range(3):
        result = group.place(object_name, place_locations)
        if result == 1:
            print(f"成功放置 {object_name}")
            return True
        print(f"放置失败，错误代码: {result}。重试中...")
        rospy.sleep(0.5)

    print(f"放置 {object_name} 失败")
    scene.remove_attached_object(group.get_end_effector_link(), name=object_name)
    scene.remove_world_object(object_name)
    return False


def move_objects_to_table_with_cache(group, scene, cache, objects_info):
    """使用缓存的物体搬运功能"""
    successful_moves = 0
    sorted_objects = sorted(
        objects_info.keys(),
        key=lambda name: math.sqrt(objects_info[name]['ground_pos'][0]**2 + objects_info[name]['ground_pos'][1]**2),
        reverse=True
    )

    for i, object_name in enumerate(sorted_objects):
        info = objects_info[object_name]
        ground_pos = info['ground_pos']
        table_pos = info['table_pos']
        
        print("\n" + "="*30)
        print(f"开始处理物体: {object_name} ({i+1}/{len(sorted_objects)})")
        print("="*30)
        
        if i == 0:
            print("移动到 'ready' 准备姿态...")
            move_to_ready_with_cache(group, cache)
            rospy.sleep(0.5)

        if pick_object_with_cache(group, scene, cache, object_name, ground_pos[0], ground_pos[1], ground_pos[2]):
            print("抓取成功，准备放置...")
            
            # 尝试直接到放置位置的轨迹
            direct_trajectory = plan_with_cache(
                group, cache, 
                [table_pos[0], table_pos[1], table_pos[2] + 0.1], 
                "pick_to_place"
            )
            
            if direct_trajectory and execute_trajectory(group, direct_trajectory):
                print("使用直接轨迹到放置位置...")
                if place_object_with_cache(group, scene, cache, object_name, table_pos[0], table_pos[1], table_pos[2]):
                    successful_moves += 1
            else:
                print("直接轨迹失败，使用ready中间点...")
                move_to_ready_with_cache(group, cache)
                rospy.sleep(0.5)
                if place_object_with_cache(group, scene, cache, object_name, table_pos[0], table_pos[1], table_pos[2]):
                    successful_moves += 1
        else:
            print(f"未能抓取 {object_name}，跳过该物体。")
        
        rospy.sleep(1)

    print("\n" + "="*30)
    print(f"搬运完成！成功搬运了 {successful_moves}/{len(objects_info)} 个物体")
    print("="*30)
    return successful_moves


def main():
    print("初始化Panda机器人多物体搬运仿真（轨迹缓存版）...")
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("panda_multi_pick_place_cached", anonymous=True)
    scene = PlanningSceneInterface()
    group = MoveGroupCommander("panda_arm")
    
    cache = TrajectoryCache()
    
    group.set_planning_time(30.0) 
    group.set_planner_id("RRTstar")
    group.set_num_planning_attempts(10)
    group.set_max_velocity_scaling_factor(0.5)
    group.set_max_acceleration_scaling_factor(0.5)
    group.set_goal_position_tolerance(0.005)
    group.set_goal_orientation_tolerance(0.01)

    print("移动到ready位置...")
    move_to_ready_with_cache(group, cache)
    rospy.sleep(1)

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

    successful_count = move_objects_to_table_with_cache(group, scene, cache, objects_info)

    print("所有任务完成，返回初始位置...")
    move_to_ready_with_cache(group, cache)
    
    print(f"最终成功率: {successful_count}/{len(objects_info)}")
    print("\n轨迹缓存统计:")
    cache_files = [f for f in os.listdir(cache.cache_dir) if f.endswith('.pkl')]
    print(f"已保存 {len(cache_files)} 个轨迹文件")
    
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