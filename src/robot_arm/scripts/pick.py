#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
import tf.transformations as tf
from moveit_commander import MoveGroupCommander, PlanningSceneInterface
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time
import math
import pickle
import os
import rospkg
import numpy as np

class CoordinateSystem:
    """统一的坐标系管理类（使用原始脚本的正确坐标）"""
    def __init__(self, base_frame="panda_link0"):
        self.base_frame = base_frame
        # 工作空间 - 扩展X范围以包含桌面放置位置
        self.workspace = {
            'x_range': [-0.2, 0.8],  # 扩展到-0.2以包含-0.08的桌面位置
            'y_range': [-0.6, 0.6],
            'z_range': [0.0, 0.6],   # 稍微增加Z范围
        }
        # 关键高度
        self.key_positions = {
            'ground_level': 0.025,
            'table_surface': 0.4,
            'approach_offset': 0.15,
            'safe_height': 0.3,
        }
        # 场景固定物体（使用原始脚本的正确坐标）
        self.scene_objects = {
            'table': {
                'position': [0.0, 0.5, 0.2],  # 原始桌子位置
                'size': [0.6, 0.4, 0.4],
                'description': '目标桌子'
            },
            'ground': {
                'position': [0.0, 0.0, -0.05],
                'size': [2.0, 2.0, 0.1],
                'description': '地面'
            }
        }
        
        # 使用原始脚本的精确坐标配置
        self.objects_config = {
            "object_0_0": {"ground_pos": [0.45, -0.15, 0.025], "table_pos": [-0.08, 0.48, 0.40], "size": [0.04, 0.04, 0.06]},
            "object_0_1": {"ground_pos": [0.45, -0.05, 0.025], "table_pos": [-0.08, 0.60, 0.40], "size": [0.04, 0.04, 0.06]},
            "object_1_0": {"ground_pos": [0.55, -0.15, 0.025], "table_pos": [0.04, 0.48, 0.40], "size": [0.04, 0.04, 0.06]},
            "object_1_1": {"ground_pos": [0.55, -0.05, 0.025], "table_pos": [0.04, 0.60, 0.40], "size": [0.04, 0.04, 0.06]},
        }

    def create_pose(self, position, orientation=None):
        """创建标准的Pose消息（修复方向，去除45度旋转）"""
        pose = geometry_msgs.msg.Pose()
        pose.position.x = position[0]
        pose.position.y = position[1] 
        pose.position.z = position[2]
        
        if orientation is None:
            # 使用原始脚本的正确方向：绕X轴旋转180度（垂直向下），无Z轴旋转
            quat = tf.quaternion_from_euler(math.pi, 0, 0)  # 移除math.pi/4旋转
        else:
            quat = orientation
            
        pose.orientation = geometry_msgs.msg.Quaternion(*quat)
        return pose
    
    def create_pose_stamped(self, position, orientation=None):
        """创建带时间戳的Pose消息"""
        pose_stamped = geometry_msgs.msg.PoseStamped()
        pose_stamped.header.frame_id = self.base_frame
        pose_stamped.pose = self.create_pose(position, orientation)
        return pose_stamped
    
    def get_approach_pose(self, target_position):
        """获取接近位置"""
        approach_pos = [
            target_position[0],
            target_position[1],
            target_position[2] + self.key_positions['approach_offset']
        ]
        return self.create_pose(approach_pos)
    
    def get_grasp_pose(self, target_position):
        """获取抓取位置"""
        grasp_pos = [
            target_position[0],
            target_position[1], 
            target_position[2] + 0.1  # 稍微高于物体
        ]
        return self.create_pose(grasp_pos)
    
    def get_retreat_pose(self, target_position):
        """获取撤退位置"""
        retreat_pos = [
            target_position[0],
            target_position[1],
            target_position[2] + self.key_positions['safe_height']
        ]
        return self.create_pose(retreat_pos)
    
    def validate_position(self, position):
        """验证位置是否在工作空间内"""
        x, y, z = position
        ws = self.workspace
        
        if not (ws['x_range'][0] <= x <= ws['x_range'][1]):
            return False, f"X坐标 {x} 超出范围 {ws['x_range']}"
        if not (ws['y_range'][0] <= y <= ws['y_range'][1]):
            return False, f"Y坐标 {y} 超出范围 {ws['y_range']}"
        if not (ws['z_range'][0] <= z <= ws['z_range'][1]):
            return False, f"Z坐标 {z} 超出范围 {ws['z_range']}"
            
        return True, "位置有效"
    
    def print_coordinates_info(self):
        """打印坐标系统信息（更新为原始坐标）"""
        print("\n" + "="*50)
        print("坐标系统信息（使用原始脚本坐标）")
        print("="*50)
        print(f"基座坐标系: {self.base_frame}")
        print(f"工作空间: X{self.workspace['x_range']} Y{self.workspace['y_range']} Z{self.workspace['z_range']}")
        print(f"桌子位置: {self.scene_objects['table']['position']}")
        print(f"桌子尺寸: {self.scene_objects['table']['size']}")
        print("\n物体配置 (ground -> table):")
        for name, cfg in self.objects_config.items():
            print(f"  {name}: {cfg['ground_pos']} -> {cfg['table_pos']}")
        print("\n关键高度:")
        for k, v in self.key_positions.items():
            print(f"  {k}: {v}")
        print("="*50 + "\n")


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
        
        self.joint_positions_cache = {}
    
    def save_joint_values(self, joint_values, name):
        """保存关节值"""
        self.joint_positions_cache[name] = list(joint_values)
        cache_file = os.path.join(self.cache_dir, f"joints_{name}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(joint_values, f)
            return True
        except Exception as e:
            print(f"保存关节位置失败: {e}")
            return False
    
    def load_joint_values(self, name):
        """加载关节值"""
        if name in self.joint_positions_cache:
            return self.joint_positions_cache[name]
        
        cache_file = os.path.join(self.cache_dir, f"joints_{name}.pkl")
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                joint_values = pickle.load(f)
            self.joint_positions_cache[name] = joint_values
            return joint_values
        except Exception as e:
            print(f"加载关节位置失败: {e}")
            return None


class PTPController:
    """点到点（PTP）控制器"""
    def __init__(self, group, cache, coord_system):
        self.group = group
        self.cache = cache
        self.coord_system = coord_system
        
        # 优化的规划参数
        self.group.set_planning_time(2.0)
        self.group.set_planner_id("RRTConnect")
        self.group.set_num_planning_attempts(1)
        self.group.set_max_velocity_scaling_factor(0.6)
        self.group.set_max_acceleration_scaling_factor(0.6)
        self.group.set_goal_joint_tolerance(0.001)
        self.group.set_goal_position_tolerance(0.001)
        self.group.set_goal_orientation_tolerance(0.001)
        
        # 预定义的安全关节位置
        self.safe_positions = {
            'home': [0, -0.785, 0, -2.356, 0, 1.571, 0.785],
            'ready': [0, -0.5, 0, -2.0, 0, 1.5, 0.785],
        }
    
    def move_to_joint_position(self, joint_values, wait=True):
        """移动到关节位置"""
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
        """移动到预定义位置"""
        cached_joints = self.cache.load_joint_values(name)
        if cached_joints:
            return self.move_to_joint_position(cached_joints, wait)
        
        if name in self.safe_positions:
            joints = self.safe_positions[name]
            if self.move_to_joint_position(joints, wait):
                self.cache.save_joint_values(joints, name)
                return True
        
        try:
            self.group.set_named_target(name)
            success = self.group.go(wait=wait)
            if success and wait:
                current_joints = self.group.get_current_joint_values()
                self.cache.save_joint_values(current_joints, name)
            return success
        except:
            return False
    
    def move_to_pose_ptp(self, target_pose, wait=True):
        """移动到目标位姿"""
        try:
            self.group.set_pose_target(target_pose)
            success = self.group.go(wait=wait)
            if wait:
                self.group.stop()
                self.group.clear_pose_targets()
            return success
        except Exception as e:
            print(f"移动到目标位姿失败: {e}")
            return False


def add_collision_objects(scene, coord_system):
    """添加场景物体（使用原始脚本的正确配置）"""
    rospy.sleep(1)
    
    # 地面
    ground_pose = geometry_msgs.msg.PoseStamped()
    ground_pose.header.frame_id = "panda_link0"
    ground_pose.pose.position.z = -0.05
    ground_pose.pose.orientation.w = 1.0
    scene.add_box("ground", ground_pose, size=(2.0, 2.0, 0.1))
    print("添加地面: [0.0, 0.0, -0.05]")

    # 目标桌子 - 使用原始位置
    table_pose = geometry_msgs.msg.PoseStamped()
    table_pose.header.frame_id = "panda_link0"
    table_pose.pose.position.x = 0.0
    table_pose.pose.position.y = 0.5
    table_pose.pose.position.z = 0.2
    table_pose.pose.orientation.w = 1.0
    scene.add_box("target_table", table_pose, size=(0.6, 0.4, 0.4))
    print("添加目标桌子: [0.0, 0.5, 0.2]")

    # 添加可移动物体
    for name, config in coord_system.objects_config.items():
        object_pose = geometry_msgs.msg.PoseStamped()
        object_pose.header.frame_id = "panda_link0"
        pos = config['ground_pos']
        object_pose.pose.position.x = pos[0]
        object_pose.pose.position.y = pos[1]
        object_pose.pose.position.z = pos[2]
        object_pose.pose.orientation.w = 1.0
        scene.add_box(name, object_pose, size=config['size'])
        print(f"添加物体 {name}: {config['ground_pos']}")

    rospy.sleep(2)


def control_gripper(action="open"):
    """控制夹爪"""
    try:
        gripper_group = MoveGroupCommander("panda_hand")
        gripper_group.set_planning_time(1.0)
        
        if action == "open":
            gripper_group.set_joint_value_target([0.04, 0.04])
        else:  # close
            gripper_group.set_joint_value_target([0.01, 0.01])
            
        return gripper_group.go(wait=True)
    except:
        print(f"警告: 无法{action}夹爪")
        return False


def pick_object(ptp_controller, group, scene, object_name, coord_system):
    """抓取物体（使用原始脚本的正确方向）"""
    config = coord_system.objects_config[object_name]
    target_pos = config['ground_pos']
    
    print(f"抓取 {object_name} 位置: {target_pos}")
    
    # 验证位置
    valid, msg = coord_system.validate_position(target_pos)
    if not valid:
        print(f"位置验证失败: {msg}")
        return False
    
    # 1. 打开夹爪
    control_gripper("open")
    
    # 2. 移动到接近位置
    approach_pose = coord_system.get_approach_pose(target_pos)
    if not ptp_controller.move_to_pose_ptp(approach_pose):
        print("无法到达接近位置")
        return False
    
    rospy.sleep(0.3)
    
    # 3. 移动到抓取位置
    grasp_pose = coord_system.get_grasp_pose(target_pos)
    if not ptp_controller.move_to_pose_ptp(grasp_pose):
        print("无法到达抓取位置")
        return False
    
    rospy.sleep(0.3)
    
    # 4. 关闭夹爪
    control_gripper("close")
    rospy.sleep(0.5)
    
    # 5. 附着物体
    scene.attach_box(group.get_end_effector_link(), object_name)
    rospy.sleep(0.3)
    
    # 6. 提升
    retreat_pose = coord_system.get_retreat_pose(target_pos)
    if ptp_controller.move_to_pose_ptp(retreat_pose):
        print(f"✓ 成功抓取 {object_name}")
        return True
    else:
        scene.remove_attached_object(group.get_end_effector_link(), name=object_name)
        print(f"✗ 提升失败")
        return False


def place_object(ptp_controller, group, scene, object_name, coord_system):
    """放置物体（使用原始脚本的正确坐标和方向）"""
    config = coord_system.objects_config[object_name]
    target_pos = config['table_pos']
    
    print(f"放置 {object_name} 位置: {target_pos}")
    
    # 验证位置
    valid, msg = coord_system.validate_position(target_pos)
    if not valid:
        print(f"位置验证失败: {msg}")
        return False
    
    # 1. 移动到放置接近位置
    approach_pose = coord_system.get_approach_pose(target_pos)
    if not ptp_controller.move_to_pose_ptp(approach_pose):
        print("无法到达放置接近位置")
        return False
    
    rospy.sleep(0.3)
    
    # 2. 下降到放置位置（调整为桌面上方合适高度）
    place_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.13]  # 使用原始脚本的放置高度
    place_pose = coord_system.create_pose(place_pos)
    if not ptp_controller.move_to_pose_ptp(place_pose):
        print("无法到达放置位置")
        return False
    
    rospy.sleep(0.3)
    
    # 3. 解除附着
    scene.remove_attached_object(group.get_end_effector_link(), name=object_name)
    rospy.sleep(0.3)
    
    # 4. 打开夹爪
    control_gripper("open")
    rospy.sleep(0.5)
    
    # 5. 撤退
    retreat_pose = coord_system.get_retreat_pose(target_pos)
    if ptp_controller.move_to_pose_ptp(retreat_pose):
        print(f"✓ 成功放置 {object_name}")
        return True
    else:
        print(f"✓ 放置成功（撤退有警告）")
        return True


def main():
    print("\nPanda机器人PTP控制多物体搬运系统\n")
    
    # 初始化ROS和MoveIt
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("panda_ptp_pick_place", anonymous=True)
    
    scene = PlanningSceneInterface()
    group = MoveGroupCommander("panda_arm")
    
    # 创建坐标系统和控制器
    coord_system = CoordinateSystem()
    cache = JointTrajectoryCache()
    ptp_controller = PTPController(group, cache, coord_system)
    
    # 打印坐标系统信息
    coord_system.print_coordinates_info()
    
    # 设置场景
    print("初始化场景...")
    scene.clear()
    rospy.sleep(1)
    
    # 清理附着物体
    attached_objects = scene.get_attached_objects()
    for obj in attached_objects.keys():
        scene.remove_attached_object(group.get_end_effector_link(), name=obj)
        rospy.sleep(0.3)
    
    add_collision_objects(scene, coord_system)
    
    # 移动到初始位置
    print("移动到初始位置...")
    ptp_controller.move_to_named_position('home')
    rospy.sleep(1)
    
    # 开始搬运
    print("\n开始搬运作业...\n")
    start_time = time.time()
    successful_count = 0
    
    object_names = list(coord_system.objects_config.keys())
    
    for i, object_name in enumerate(object_names):
        print(f"\n处理物体: {object_name} ({i+1}/{len(object_names)})")
        
        # 移动到准备位置
        ptp_controller.move_to_named_position('ready')
        rospy.sleep(0.5)
        
        # 抓取和放置
        if pick_object(ptp_controller, group, scene, object_name, coord_system):
            if place_object(ptp_controller, group, scene, object_name, coord_system):
                successful_count += 1
            else:
                # 清理失败的附着物体
                scene.remove_attached_object(group.get_end_effector_link(), name=object_name)
        
        rospy.sleep(0.5)
    
    elapsed_time = time.time() - start_time
    
    # 返回home位置
    print("\n返回home位置...")
    ptp_controller.move_to_named_position('home')
    
    # 打印结果
    print(f"\n搬运完成！成功率: {successful_count}/{len(object_names)} ({successful_count*100/len(object_names):.1f}%)")
    print(f"总用时: {elapsed_time:.1f} 秒")
    
    rospy.sleep(2)
    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        print("\n程序被中断")
    except Exception as e:
        print(f"\n程序错误: {e}")