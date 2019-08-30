import rospy
from std_msgs.msg import Float64MultiArray
from meta_mb.envs.pr2.real_pr2_reach_env import PR2ReacherEnv

import numpy as np
from subprocess import Popen, check_output

class RealPR2(PR2ReacherEnv):
	def __init__(self, **kwargs):
        super(RealPR2, self).__init__(kwargs)
		rospy.init_node('rllab_torque_publisher')
		rospy.set_param('simple_torque_controller/SimpleTorqueController/type', 'simple_torque_controller/SimpleTorqueController')
        output = check_output("rosrun pr2_controller_manager pr2_controller_manager list", shell=True)
        output_lines = output.split('\n')
        to_kill_l_arm = False
        to_load_controller = True
        to_start_controller = True
        for line in output_lines: 
            if 'l_arm_controller' in line:
                print line
                to_kill_l_arm = True
            if 'simple_torque_controller' in line:
                print line
                to_load_controller = False
                if 'running' in line:
                    to_start_controller = False
        if to_kill_l_arm:
            kill_l_arm = Popen("rosrun pr2_controller_manager pr2_controller_manager " \
                    + "kill l_arm_controller", shell=True)
            rospy.sleep(0.2)
        if to_load_controller:
            load_simple_torque = Popen("rosrun pr2_controller_manager pr2_controller_manager " \
                    + "load simple_torque_controller/SimpleTorqueController", shell=True)
            rospy.sleep(0.2)
        if to_start_controller:
            start_simple_torque = Popen("rosrun pr2_controller_manager pr2_controller_manager " \
                    + "start simple_torque_controller/SimpleTorqueController", shell=True)
            rospy.sleep(0.2)

        
        self.torque_array_pub = rospy.Publisher('/rllab_torque', Float64MultiArray)

        rospy.Subscriber("/rllab_obs", Float64MultiArray, self.obs_callback)

