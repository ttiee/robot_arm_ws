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
import numpy as np

tau = 2 * math.pi

class JointTrajectoryCache:
    """关节空间轨迹缓存管理类"""
    def __init__(self, cache_dir=None):
        if cache_dir is None:
            rospack = rospkg.RosPack()
            try:
                package_path = rospack.get_path('robot_arm')
                cache_dir = os.path.join(package_path, 'joint_trajectory_cache')
            except rospkg.ResourceNotFound:
                cache_dir = "joint_trajectory_cache"
        
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        print(f"关节轨迹缓存目录: {self.cache_dir}")
        
        # 预定义的关节位置缓存
        self.joint_positions_cache = {}
    
    def _get_cache_key(self, target_name, pose_data=None):
        """生成缓存键"""
        if pose_data:
            key_str = f"{target_name}_{str(pose_data)}"
        else:
            key_str = target_name
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def save_joint_values(self, joint_values, name):
        """保存关节值"""
        self.joint_positions_cache[name] = list(joint_values)
        cache_file = os.path.join(self.cache_dir, f"joints_{name}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(joint_values, f)
            print(f"关节位置已保存: {name}")
            return True
        except Exception as e:
            print(f"保存关节位置失败: {e}")
            return False
    
    def load_joint_values(self, name):
        """加载关节值"""
        # 首先检查内存缓存
        if name in self.joint_positions_cache:
            return self.joint_positions_cache[name]
        
        cache_file = os.path.join(self.cache_dir, f"joints_{name}.pkl")
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                joint_values = pickle.load(f)
            self.joint_positions_cache[name] = joint_values
            print(f"关节位置已加载: {name}")
            return joint_values
        except Exception as e:
            print(f"加载关节位置失败: {e}")
            return None


class PTPController:
    """点到点（PTP）控制器"""
    def __init__(self, group, cache):
        self.group = group
        self.cache = cache
        self.joint_names = group.get_active_joints()
        
        # 设置更激进的规划参数
        self.group.set_planning_time(2.0)  # 大幅减少规划时间避免超时
        self.group.set_planner_id("RRTConnect")  # 使用更快的规划器
        self.group.set_num_planning_attempts(1)  # 减少尝试次数
        self.group.set_max_velocity_scaling_factor(0.6)  # 适中速度
        self.group.set_max_acceleration_scaling_factor(0.6)  # 适中加速度
        self.group.set_goal_joint_tolerance(0.001)
        self.group.set_goal_position_tolerance(0.001)
        self.group.set_goal_orientation_tolerance(0.001)
        
        # 预定义的安全关节位置
        self.safe_positions = {
            'home': [0, -0.785, 0, -2.356, 0, 1.571, 0.785],
            'ready': [0, -0.5, 0, -2.0, 0, 1.5, 0.785],
            'pick_approach': [0, 0.2, 0, -1.8, 0, 2.0, 0.785],
            'place_approach': [0, -0.3, 0, -2.2, 0, 1.9, 0.785]
        }
        
    def move_to_joint_position(self, joint_values, wait=True):
        """直接移动到关节位置（PTP移动）"""
        try:
            self.group.set_joint_value_target(joint_values)
            success = self.group.go(wait=wait)
            if wait:
                self.group.stop()
                self.group.clear_pose_targets()
            return success
        except Exception as e:
            print(f"PTP移动失败: {e}")
            return False
    
    def move_to_named_position(self, name, wait=True):
        """移动到预定义的命名位置"""
        # 尝试从缓存加载
        cached_joints = self.cache.load_joint_values(name)
        if cached_joints:
            return self.move_to_joint_position(cached_joints, wait)
        
        # 使用预定义位置
        if name in self.safe_positions:
            joints = self.safe_positions[name]
            if self.move_to_joint_position(joints, wait):
                self.cache.save_joint_values(joints, name)
                return True
        
        # 使用MoveIt的命名目标
        try:
            self.group.set_named_target(name)
            success = self.group.go(wait=wait)
            if success and wait:
                current_joints = self.group.get_current_joint_values()
                self.cache.save_joint_values(current_joints, name)
            return success
        except:
            return False
    
    def compute_ik_joint_position(self, target_pose):
        """计算目标位姿的逆运动学解（关节位置）"""
        # 设置目标位姿
        self.group.set_pose_target(target_pose)
        
        # 获取规划（但不执行）
        plan = self.group.plan()
        
        if isinstance(plan, tuple):
            success, trajectory, _, _ = plan
            if success and trajectory and trajectory.joint_trajectory.points:
                # 返回轨迹终点的关节位置
                final_point = trajectory.joint_trajectory.points[-1]
                return list(final_point.positions)
        
        return None
    
    def move_cartesian(self, waypoints, wait=True):
        """笛卡尔路径规划（用于直线运动）"""
        try:
            (plan, fraction) = self.group.compute_cartesian_path(
                waypoints,
                0.01,  # eef步长
                0.0,   # 跳跃阈值
                avoid_collisions=True  # 添加明确的避碰参数
            )
            
            if fraction > 0.9:  # 如果90%以上的路径可行
                success = self.group.execute(plan, wait=wait)
                if wait:
                    self.group.stop()
                return success
            else:
                print(f"笛卡尔路径规划不完整: {fraction*100:.1f}%")
                return False
        except Exception as e:
            print(f"笛卡尔路径规划失败: {e}")
            return False
    
    def move_to_pose_ptp(self, target_pose, wait=True):
        """通过关节空间移动到目标位姿"""
        try:
            # 直接使用pose target并用go执行
            self.group.set_pose_target(target_pose)
            success = self.group.go(wait=wait)
            if wait:
                self.group.stop()
                self.group.clear_pose_targets()
            return success
        except Exception as e:
            print(f"移动到目标位姿失败: {e}")
            return False


def open_gripper(posture):
    posture.joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
    point = JointTrajectoryPoint()
    point.positions = [0.04, 0.04]
    point.time_from_start = rospy.Duration(0.5)
    posture.points = [point]


def closed_gripper(posture):
    posture.joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
    point = JointTrajectoryPoint()
    point.positions = [0.01, 0.01]  # 稍微增加抓取力
    point.time_from_start = rospy.Duration(0.5)
    posture.points = [point]


def add_collision_objects(scene, objects_info):
    """添加场景物体"""
    rospy.sleep(1)
    
    # 地面
    ground_pose = geometry_msgs.msg.PoseStamped()
    ground_pose.header.frame_id = "panda_link0"
    ground_pose.pose.position.z = -0.05
    ground_pose.pose.orientation.w = 1.0
    scene.add_box("ground", ground_pose, size=(2.0, 2.0, 0.1))

    # 目标桌子 - 调整位置使其更容易到达
    table_pose = geometry_msgs.msg.PoseStamped()
    table_pose.header.frame_id = "panda_link0"
    table_pose.pose.position.x = 0.0
    table_pose.pose.position.y = 0.5  # 稍微近一点
    table_pose.pose.position.z = 0.2
    table_pose.pose.orientation.w = 1.0
    scene.add_box("target_table", table_pose, size=(0.6, 0.4, 0.4))

    # 添加物体
    for name, info in objects_info.items():
        object_pose = geometry_msgs.msg.PoseStamped()
        object_pose.header.frame_id = "panda_link0"
        pos = info['ground_pos']
        object_pose.pose.position.x = pos[0]
        object_pose.pose.position.y = pos[1]
        object_pose.pose.position.z = pos[2]
        object_pose.pose.orientation.w = 1.0
        scene.add_box(name, object_pose, size=(0.04, 0.04, 0.06))

    rospy.sleep(2)
    return list(objects_info.keys())


def pick_with_ptp(ptp_controller, group, scene, object_name, x, y, z):
    """使用PTP控制进行抓取（简化版）"""
    print(f"使用PTP控制抓取 {object_name}...")
    
    # 直接使用手动控制方法，跳过内置pick功能
    
    # 1. 初始化夹爪（如果需要）
    try:
        gripper_group = MoveGroupCommander("panda_hand")
        gripper_group.set_planning_time(1.0)  # 减少夹爪规划时间
    except:
        print("警告: 无法初始化夹爪控制组，尝试继续...")
        gripper_group = None
    
    # 2. 打开夹爪
    if gripper_group:
        try:
            gripper_group.set_named_target("open")
            gripper_group.go(wait=True)
        except:
            print("警告: 无法打开夹爪，尝试继续...")
    
    # 3. 移动到接近位置
    approach_pose = geometry_msgs.msg.Pose()
    approach_pose.position.x = x
    approach_pose.position.y = y
    approach_pose.position.z = z + 0.2  # 接近高度
    # 修正方向：绕X轴旋转180度，Z轴旋转45度对齐方块
    quat = tf.quaternion_from_euler(-math.pi, 0, math.pi/4)
    approach_pose.orientation = geometry_msgs.msg.Quaternion(*quat)
    
    if not ptp_controller.move_to_pose_ptp(approach_pose):
        print(f"无法到达接近位置")
        return False
    
    print("到达接近位置")
    rospy.sleep(0.3)
    
    # 4. 下降到抓取位置（使用更小的步进）
    grasp_pose = geometry_msgs.msg.Pose()
    grasp_pose.position.x = x
    grasp_pose.position.y = y
    grasp_pose.position.z = z + 0.1  # 降低抓取高度
    grasp_pose.orientation = approach_pose.orientation
    
    if not ptp_controller.move_to_pose_ptp(grasp_pose):
        print(f"无法到达抓取位置")
        return False
    
    print("到达抓取位置")
    rospy.sleep(0.3)
    
    # 5. 关闭夹爪抓取物体
    if gripper_group:
        try:
            gripper_group.set_named_target("close")
            gripper_group.go(wait=True)
        except:
            # 尝试直接设置关节值
            try:
                gripper_group.set_joint_value_target([0.01, 0.01])
                gripper_group.go(wait=True)
            except:
                print("警告: 无法关闭夹爪")
    
    rospy.sleep(0.5)  # 等待夹爪稳定
    
    # 6. 附着物体到夹爪
    touch_links = group.get_end_effector_link()
    scene.attach_box(touch_links, object_name)
    rospy.sleep(0.3)
    
    # 7. 提升物体
    lift_pose = geometry_msgs.msg.Pose()
    lift_pose.position.x = x
    lift_pose.position.y = y
    lift_pose.position.z = z + 0.25
    lift_pose.orientation = approach_pose.orientation

    if ptp_controller.move_to_pose_ptp(lift_pose):
        print(f"✓ 成功抓取并提升 {object_name}")
        return True
    else:
        # 如果提升失败，尝试清理
        scene.remove_attached_object(touch_links, name=object_name)
        print(f"✗ 提升 {object_name} 失败")
        return False


def place_with_ptp(ptp_controller, group, scene, object_name, x, y, z):
    """使用PTP控制进行放置（简化版）"""
    print(f"使用PTP控制放置 {object_name}...")
    
    # 直接使用手动控制方法
    
    # 初始化夹爪（如果需要）
    try:
        gripper_group = MoveGroupCommander("panda_hand")
        gripper_group.set_planning_time(1.0)
    except:
        print("警告: 无法初始化夹爪控制组，尝试继续...")
        gripper_group = None
    
    # 1. 移动到放置接近位置
    place_approach_pose = geometry_msgs.msg.Pose()
    place_approach_pose.position.x = x
    place_approach_pose.position.y = y
    place_approach_pose.position.z = z + 0.15  # 放置接近高度
    # 使用垂直向下的方向，与抓取保持一致
    quat = tf.quaternion_from_euler(math.pi, 0, 0)  # 末端执行器垂直向下
    place_approach_pose.orientation = geometry_msgs.msg.Quaternion(*quat)
    
    if not ptp_controller.move_to_pose_ptp(place_approach_pose):
        print(f"无法到达放置接近位置")
        # 清理附着的物体
        scene.remove_attached_object(group.get_end_effector_link(), name=object_name)
        return False
    
    print("到达放置接近位置")
    rospy.sleep(0.3)
    
    # 2. 下降到放置位置
    place_pose = geometry_msgs.msg.Pose()
    place_pose.position.x = x
    place_pose.position.y = y
    place_pose.position.z = z + 0.13  # 放置高度
    # 保持与接近位置相同的方向
    place_pose.orientation = place_approach_pose.orientation  # 保持相同方向
    
    if not ptp_controller.move_to_pose_ptp(place_pose):
        print(f"无法到达放置位置")
        scene.remove_attached_object(group.get_end_effector_link(), name=object_name)
        return False
    
    print("到达放置位置")
    rospy.sleep(0.3)
    
    # 3. 解除物体附着（在打开夹爪前）
    touch_links = group.get_end_effector_link()
    scene.remove_attached_object(touch_links, name=object_name)
    rospy.sleep(0.3)
    
    # 4. 打开夹爪释放物体
    if gripper_group:
        try:
            gripper_group.set_named_target("open")
            gripper_group.go(wait=True)
        except:
            try:
                gripper_group.set_joint_value_target([0.04, 0.04])
                gripper_group.go(wait=True)
            except:
                print("警告: 无法打开夹爪")
    
    rospy.sleep(0.5)  # 等待物体稳定
    
    # 5. 提升夹爪撤退
    retreat_pose = geometry_msgs.msg.Pose()
    retreat_pose.position.x = x
    retreat_pose.position.y = y
    retreat_pose.position.z = z + 0.25  # 撤退高度
    retreat_pose.orientation = place_approach_pose.orientation  # 保持相同方向
    
    if ptp_controller.move_to_pose_ptp(retreat_pose):
        print(f"✓ 成功放置 {object_name}")
        return True
    else:
        # 即使撤退失败，物体已经放置
        print(f"✓ 放置 {object_name} 成功（撤退时有警告）")
        return True


def move_objects_with_ptp(ptp_controller, group, scene, objects_info):
    """使用PTP控制搬运物体"""
    successful_moves = 0
    
    # 按距离排序物体（从近到远）
    sorted_objects = sorted(
        objects_info.keys(),
        key=lambda name: math.sqrt(
            objects_info[name]['ground_pos'][0]**2 + 
            objects_info[name]['ground_pos'][1]**2
        )
    )

    for i, object_name in enumerate(sorted_objects):
        info = objects_info[object_name]
        ground_pos = info['ground_pos']
        table_pos = info['table_pos']
        
        print("\n" + "="*30)
        print(f"处理物体: {object_name} ({i+1}/{len(sorted_objects)})")
        print("="*30)
        
        # 移动到准备位置
        if i == 0 or successful_moves % 2 == 0:
            print("移动到ready位置...")
            ptp_controller.move_to_named_position('ready')
            rospy.sleep(0.5)
        
        # 抓取
        if pick_with_ptp(ptp_controller, group, scene, object_name, 
                        ground_pos[0], ground_pos[1], ground_pos[2]):
            
            # 提升到安全高度
            ptp_controller.move_to_named_position('place_approach')
            rospy.sleep(0.3)
            
            # 放置
            if place_with_ptp(ptp_controller, group, scene, object_name,
                            table_pos[0], table_pos[1], table_pos[2]):
                successful_moves += 1
                print(f"✓ {object_name} 搬运成功")
            else:
                print(f"✗ {object_name} 放置失败")
                # 清理附着的物体
                scene.remove_attached_object(group.get_end_effector_link(), name=object_name)
        else:
            print(f"✗ {object_name} 抓取失败")
            scene.remove_world_object(object_name)
        
        rospy.sleep(0.5)

    return successful_moves


def main():
    print("\n" + "="*50)
    print("Panda机器人PTP控制多物体搬运系统")
    print("="*50 + "\n")
    
    # 初始化ROS和MoveIt
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("panda_ptp_pick_place", anonymous=True)
    
    scene = PlanningSceneInterface()
    group = MoveGroupCommander("panda_arm")
    
    # 创建缓存和PTP控制器
    cache = JointTrajectoryCache()
    ptp_controller = PTPController(group, cache)
    
    # 优化后的物体位置（更合理的工作空间布局）
    objects_info = {
        "object_0_0": {"ground_pos": [0.45, -0.15, 0.025], "table_pos": [-0.08, 0.48, 0.40]},  # 调整桌面高度
        "object_0_1": {"ground_pos": [0.45, -0.05, 0.025], "table_pos": [-0.08, 0.60, 0.40]},
        "object_1_0": {"ground_pos": [0.55, -0.15, 0.025], "table_pos": [0.04, 0.48, 0.40]},
        "object_1_1": {"ground_pos": [0.55, -0.05, 0.025], "table_pos": [0.04, 0.60, 0.40]},
    }
    
    # 设置场景
    print("初始化场景...")
    scene.clear()
    rospy.sleep(1)
    
    # 清理可能的附着物体
    attached_objects = scene.get_attached_objects()
    for obj in attached_objects.keys():
        scene.remove_attached_object(group.get_end_effector_link(), name=obj)
        rospy.sleep(0.3)
    
    # 添加场景物体
    add_collision_objects(scene, objects_info)
    
    # 移动到初始位置
    print("移动到初始位置...")
    ptp_controller.move_to_named_position('home')
    rospy.sleep(1)
    
    # 开始搬运
    print("\n开始搬运作业...\n")
    start_time = time.time()
    
    successful_count = move_objects_with_ptp(ptp_controller, group, scene, objects_info)
    
    elapsed_time = time.time() - start_time
    
    # 返回home位置
    print("\n返回home位置...")
    ptp_controller.move_to_named_position('home')
    
    # 打印统计信息
    print("\n" + "="*50)
    print("搬运作业完成！")
    print(f"成功率: {successful_count}/{len(objects_info)} ({successful_count*100/len(objects_info):.1f}%)")
    print(f"总用时: {elapsed_time:.1f} 秒")
    print(f"平均每个物体: {elapsed_time/len(objects_info):.1f} 秒")
    
    # 缓存统计
    cache_files = [f for f in os.listdir(cache.cache_dir) if f.endswith('.pkl')]
    print(f"\n缓存的关节位置: {len(cache_files)} 个")
    print("="*50)
    
    rospy.sleep(5)
    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        print("\n程序被中断")
    except Exception as e:
        import traceback
        print(f"\n程序错误: {e}")
        traceback.print_exc()