import random
from threading import Thread
import math
import rospy
import time
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist


class Darwin:
    """
    Client ROS class for manipulating Darwin OP in Gazebo
    """

    def __init__(self, ns="/darwin/"):
        self.ns = ns
        self.joints = None
        self.angles = None
        self._joint_velocities = None
        self._joint_positions = None

        self._sub_joints = rospy.Subscriber(ns + "joint_states", JointState, self._cb_joints, queue_size=1)
        rospy.loginfo("Waiting for joints to be populated...")
        while not rospy.is_shutdown():
            if self.joints is not None: break
            rospy.sleep(0.1)
            rospy.loginfo("Waiting for joints to be populated...")
        rospy.loginfo("Joints populated")

        rospy.loginfo("Creating joint command publishers")
        self._pub_joints = {}
        self._sub_joints = {}
        for j in self.joints:
            p = rospy.Publisher(self.ns + j + "_position_controller/command", Float64)
            s = rospy.Subscriber(self.ns + j + "_position_controller/state", Float64, self._joint_state_callback)
            self._pub_joints[j] = p

        rospy.sleep(1)

        self.joint_names = [
            'j_pan',
            'j_tilt',
            'j_shoulder_r',
            'j_high_arm_r',
            'j_low_arm_r',
            'j_wrist_r',
            'j_gripper_r',
            'j_pelvis_r',
            'j_thigh1_r',
            'j_thigh2_r',
            'j_tibia_r',
            'j_ankle1_r',
            'j_ankle2_r',
            'j_shoulder_l',
            'j_high_arm_l',
            'j_low_arm_l',
            'j_wrist_l',
            'j_gripper_l',
            'j_pelvis_l',
            'j_thigh1_l',
            'j_thigh2_l',
            'j_tibia_l',
            'j_ankle1_l',
            'j_ankle2_l'
        ]

        self._pub_cmd_vel = rospy.Publisher(ns + "cmd_vel", Twist)

    def set_walk_velocity(self, x, y, t):
        msg = Twist()
        msg.linear.x = x
        msg.linear.y = y
        msg.angular.z = t
        self._pub_cmd_vel.publish(msg)

    def _cb_joints(self, msg):
        if self.joints is None:
            self.joints = msg.name
        self.angles = msg.position

    def get_angles(self):
        if self.joints is None: return None
        if self.angles is None: return None
        return dict(zip(self.joints, self.angles))

    def my_set_angles(self):
        for joint in self._pub_joints.keys():
            msg = Float64()
            msg.data = np.random.uniform(low=-0.25, high=0.25)
            self._pub_joints[joint].publish(msg)

    def set_angles(self, angles):
        for j, v in angles.items():
            if j not in self.joints:
                rospy.logerror("Invalid joint name " + j)
                continue
            self._pub_joints[j].publish(v)

    def set_angles_slow(self, stop_angles, delay=2):
        start_angles = self.get_angles()
        start = time.time()
        stop = start + delay
        r = rospy.Rate(100)
        while not rospy.is_shutdown():
            t = time.time()
            if t > stop: break
            ratio = (t - start) / delay
            angles = interpolate(stop_angles, start_angles, ratio)
            self.set_angles(angles)
            r.sleep()

    def _joint_state_callback(self, message):
        joint_positions_temp = []
        joint_velocities_temp = []

        for name in self._joint_names:
            if name not in message["name"]:
                continue
            else:
                self.index = message["name"].index(name)
                joint_positions_temp.append(message["position"][self.index])
                joint_velocities_temp.append(message["velocity"][self.index])

        if self._gripper_joint_name in message["name"]:
            index = message["name"].index(self._gripper_joint_name)
            self._gripper_position = message["position"][index]
            self._gripper_effort = message["effort"][index]

        if len(joint_positions_temp) != 0:
            self._joint_positions = np.array(joint_positions_temp)
        if len(joint_velocities_temp) != 0:
            self._joint_velocities = np.array(joint_velocities_temp)

def interpolate(anglesa, anglesb, coefa):
    z = {}
    joints = anglesa.keys()
    for j in joints:
        z[j] = anglesa[j] * coefa + anglesb[j] * (1 - coefa)
    return z


def get_distance(anglesa, anglesb):
    d = 0
    joints = anglesa.keys()
    if len(joints) == 0: return 0
    for j in joints:
        d += abs(anglesb[j] - anglesa[j])
    d /= len(joints)
    return d
