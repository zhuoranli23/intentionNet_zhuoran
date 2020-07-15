"""
Run the learned model to control the Turtlebot in the Stage Simulator
"""
import pygame
import time
import sys
import fire
import string
import random
import numpy as np

# import local file
from joy_teleop import JOY_MAPPING
from policy import Policy
# ros packages
import rospy
from sensor_msgs.msg import Joy, Image, Imu, CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32, Float32, String
from nav_msgs.msg import Odometry
from keyboard_control.msg import Keyboard
import cv2
from cv_bridge import CvBridge

import sys
sys.path.append('/mnt/intention_net')
sys.path.append('../utils')
sys.path.append('../')
#from undistort import undistort, FRONT_CAMERA_INFO
from dataset import PioneerDataset as Dataset

# SCREEN SCALE IS FOR high dpi screen, i.e. 4K screen
SCREEN_SCALE = 1
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768

class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time

class Controller(object):
    tele_twist = Twist()
    def __init__(self, mode, scale_x, scale_z, rate):
        self._mode = mode
        self._scale_x = scale_x
        self._scale_z = scale_z
        self._timer = Timer()
        self._rate = rospy.Rate(rate)
        self._enable_auto_control = False
        self.current_control = None
        self.trajectory_index = None
        self.bridge = CvBridge()

        # Callback data store
        self.image = None
        self.intention = None
        # self.imu = None
        # self.odom = None
        # self.labeled_control = None
        self.key = None
        # self.scan = None

        # Controller mode setup: Training or Testing
        self.training = False


        # Subscribe ros messages
        rospy.Subscriber('/image', Image, self.cb_image, queue_size=1, buff_size=2 ** 10)
        # rospy.Subscriber('/scan', LaserScan, self.cb_scan, queue_size=1, buff_size=2 ** 10)
        # rospy.Subscriber('/imu', Imu, self.cb_imu, queue_size=1, buff_size=2 ** 10)
        # rospy.Subscriber('/odom', Odometry, self.cb_odom, queue_size=1, buff_size=2 ** 10)
        #rospy.Subscriber('/joy', Joy, self.cb_joy)
        rospy.Subscriber('/keyboard_control', Keyboard, self.cb_keyboard)
        # rospy.Subscriber('/labeled_control', Twist, self.cb_labeled_control, queue_size=1)
        # rospy.Subscriber('/speed', Float32, self.cb_speed, queue_size=1)

        if self._mode == "DLM":
            rospy.Subscriber('/test_intention', String, self.cb_dlm_intention, queue_size=1)
        else:
            rospy.Subscriber('/intention_lpe', Image, self.cb_lpe_intention, queue_size=1, buff_size=2 ** 10)

        # Publish Control
        self.control_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=1)

        # Publish as training data
        self.pub_intention = rospy.Publisher('/train/intention', String, queue_size=1)
        # self.pub_trajectory_index = rospy.Publisher('/train/trajectory_index', String, queue_size=1)
        self.pub_teleop_vel = rospy.Publisher('/train/mobile_base/commands/velocity', Twist, queue_size=1)
        self.pub_image = rospy.Publisher('/train/image', Image, queue_size=1)


    def cb_image(self, msg):
        self.image = msg


    #def cb_scan(self, msg):
    #    self.scan = msg

    #def cb_imu(self, msg):
    #    self.imu = msg

    #def cb_speed(self, msg):
    #    self.speed = 00.01

    #def cb_odom(self, msg):
    #    self.odom = msg


    def cb_lpe_intention(self, msg):
        self.intention = cv2.resize(CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8'), (224, 224))

    def cb_dlm_intention(self, msg):
        self.intention = msg.data

    #def cb_labeled_control(self, msg):
    #    self.labeled_control = msg

    def cb_keyboard(self, msg):
        self.tele_twist.linear.x = self._scale_x * msg.twist.linear.x
        self.tele_twist.angular.z = self._scale_z * msg.twist.angular.z

        # parse control key
        if msg.command.data == 'a':
            self._enable_auto_control = True
            print('Auto control')
        if msg.command.data == 'd':
            self._enable_auto_control = False
            print('Manual control')
        if msg.command.data == 'q':
            self.key = 'quit'

        # toggle recording mode and generate trajectory index if start recording
        if msg.command.data == 't':
            self.key = 'toggle_between_train_or_not'
            print('toggle training mode to: %s' % (not self.training))

        if msg.command.data == 's':
            self.key = 'stop'
            print('stop')

    def _random_string(self, n):
        chars = string.ascii_letters+string.digits
        ret = ''.join(random.choice(chars) for _ in range(n))
        return ret

    def _on_loop(self, policy):
        """
        Logical Loop
        """
        self._timer.tick()

        # Handle commands of the Joystick
        if self.key == "quit":
            sys.exit(-1)
        if self.key == "toggle_between_train_or_not":
            self.training = not self.training
            self.key = ""
        if self.key == "stop":
            self._enable_auto_control = False
            self.key = ""
            self.training = False
            self.tele_twist.linear.x = 0
            self.tele_twist.angular.z = 0

        # Logical robot control
        if self._enable_auto_control:
            if not self.intention:
                print("estimating pose + goal...")
            elif self.intention == "stop":
                self.tele_twist.linear.x = 0
                self.tele_twist.angular.z = 0
            else:
                if self._mode == "DLM":
                    intention = Dataset.INTENTION_MAPPING[self.intention] # map intention str => int
                    print('intention: ', intention)
                if policy.input_frame == "NORMAL":
                    image = cv2.resize(CvBridge().imgmsg_to_cv2(self.image, desired_encoding='bgr8'), (224, 224))
                    # image = cv2.resize(self.image, (224, 224))
                    pred_control = policy.predict_control(image, intention, 9.99)[0]
                    self.tele_twist.linear.x = pred_control[0] * Dataset.SCALE_VEL
                    self.tele_twist.angular.z = pred_control[1] * Dataset.SCALE_STEER

        # publish to /train/* topic to record data (if in training moode)
        if self.training:
            self._publish_as_trn()

        # publish control
        self.control_pub.publish(self.tele_twist)

    def _publish_as_trn(self):
        self.pub_intention.publish(self.intention)
        self.pub_image.publish(self.image)
        self.pub_teleop_vel.publish(self.tele_twist)

    """
    def text_to_screen(self, text, color=(200, 000, 000), pos=(WINDOW_WIDTH / 2, 30), size=30):
        text = str(text)
        font = pygame.font.SysFont('Comic Sans MS', size * SCREEN_SCALE)  # pygame.font.Font(font_type, size)
        text = font.render(text, True, color)
        text_rect = text.get_rect(center=(pos[0] * SCREEN_SCALE, pos[1] * SCREEN_SCALE))
        self._display.blit(text, text_rect)


    def get_vertical_rect(self, value, pos):
        pos = (pos[0] * SCREEN_SCALE, pos[1] * SCREEN_SCALE)
        scale = 20 * SCREEN_SCALE
        if value > 0:
            return pygame.Rect((pos[0], pos[1] - value * scale), (scale, value * scale))
        else:
            return pygame.Rect(pos, (scale, -value * scale))

    def get_horizontal_rect(self, value, pos):
        pos = (pos[0] * SCREEN_SCALE, pos[1] * SCREEN_SCALE)
        scale = 20 * SCREEN_SCALE
        if value > 0:
            return pygame.Rect((pos[0] - value * scale, pos[1]), (value * scale, scale))
        else:
            return pygame.Rect(pos, (-value * scale, scale))

    def control_bar(self, pos=(WINDOW_WIDTH - 100, WINDOW_HEIGHT - 150)):
        acc_rect = self.get_vertical_rect(self.tele_twist.linear.x, pos)
        pygame.draw.rect(self._display, (0, 255, 0), acc_rect)
        steer_rect = self.get_horizontal_rect(self.tele_twist.angular.z, (pos[0], pos[1] + 110))
        pygame.draw.rect(self._display, (0, 255, 0), steer_rect)
        if self.labeled_control is not None:
            pygame.draw.rect(self._display, (255, 0, 0),
                             self.get_vertical_rect(self.labeled_control.linear.x, (pos[0] - 20, pos[1])))
                 pygame.draw.rect(self._display, (255, 0, 0),
                             self.get_horizontal_rect(self.labeled_control.angular.z, (pos[0], pos[1] + 130)))
    def _on_render(self):

        # render loop

        if self.image is not None:
            array = cv2.resize(self.image, (WINDOW_WIDTH * SCREEN_SCALE, WINDOW_HEIGHT * SCREEN_SCALE))
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (0, 0))
        if self.speed is not None:
            self.text_to_screen('Speed: {:.4f} m/s'.format(self.speed), pos=(150, WINDOW_HEIGHT - 30))
        if self.intention is not None:
            if self._mode == 'DLM':
                self.text_to_screen(self.intention)
            else:
                surface = pygame.surfarray.make_surface(self.intention.swapaxes(0, 1))
                self._display.blit(surface, (SCREEN_SCALE * (WINDOW_WIDTH - self.intention.shape[0]) / 2, 0))

        self.control_bar()
        self.text_to_screen("Auto: {}".format(self._enable_auto_control), pos=(150, WINDOW_HEIGHT - 70))

        pygame.display.flip()

    def _initialize_game(self):
        self._display = pygame.display.set_mode(
            (WINDOW_WIDTH * SCREEN_SCALE, WINDOW_HEIGHT * SCREEN_SCALE),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

    """

    def execute(self, policy):
        while True:
            self._on_loop(policy)
            self._rate.sleep()


# wrapper for fire to get command arguments
def run_wrapper(model_dir="", mode="DLM", input_frame="NORMAL", num_intentions=4, scale_x=1, scale_z=1, rate=10):
    rospy.init_node("turtlebot_controller")
    controller = Controller(mode, scale_x, scale_z, rate)
    if not model_dir:
        policy = None
    else:
        policy = Policy(mode, input_frame, 2, model_dir, num_intentions)
    controller.execute(policy)

def main():
    fire.Fire(run_wrapper)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user! Bye Bye!')
















