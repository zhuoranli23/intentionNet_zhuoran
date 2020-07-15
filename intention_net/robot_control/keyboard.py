import rospy

from geometry_msgs.msg import Twist
from std_msgs.msg import String
from keyboard_control.msg import Keyboard
import sys, select, termios, tty

msg = """
Control Your Turtlebot!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

r/f : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%

space key, k : force stop
a: turn on auto-control mode
d: turn off auto-control mode/turn on manual-control mode
q: quit the navigation system
s: stop navigation(stop auto, stop training)
t: turn on/off training mode(toggle between)
anything else : stop smoothly

CTRL-C to quit
"""

moveBindings = {
        'i':(1,0),
        'o':(1,-1),
        'j':(0,1),
        'l':(0,-1),
        'u':(1,1),
        ',':(-1,0),
        '.':(-1,1),
        'm':(-1,-1),
           }

speedBindings={
        'r':(1.1,1.1),
        'f':(.9,.9),
        'w':(1.1,1),
        'x':(.9,1),
        'e':(1,1.1),
        'c':(1,.9),
          }

def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

speed = .2
turn = 1

def vels(speed,turn):
    return "currently:\tspeed %s\tturn %s " % (speed,turn)

if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)

    rospy.init_node('turtlebot_keyboard')
    pub = rospy.Publisher('/keyboard_control', Keyboard, queue_size=5)

    x = 0
    th = 0
    status = 0
    count = 0
    acc = 0.1
    target_speed = 0
    target_turn = 0
    control_speed = 0
    control_turn = 0
    try:
        print msg
        print vels(speed,turn)
        while(1):
            string = String()
            key = getKey()
            if key == 'a':
                x = 0
                th = 0
                control_speed = 0
                control_turn = 0

                string.data = 'a'
                print ("publish auto mode commands...")
            elif key == 'd':
                x = 0
                th = 0
                control_speed = 0
                control_turn = 0

                string.data = 'd'
                print ("publish manual mode commands...")
            elif key == 'q':
                x = 0
                th = 0
                control_speed = 0
                control_turn = 0

                string.data = 'q'
                print ("publish quit system commands...")
            elif key == 's':
                x = 0
                th = 0
                control_speed = 0
                control_turn = 0

                string.data = 's'
                print ("publish stop navigation commands...")
            elif key == 't':
                x = 0
                th = 0
                control_speed = 0
                control_turn = 0

                string.data = 't'
                print ("publish toggle training mode commands")
            elif key in moveBindings.keys():
                x = moveBindings[key][0]
                th = moveBindings[key][1]
                count = 0
            elif key in speedBindings.keys():
                speed = speed * speedBindings[key][0]
                turn = turn * speedBindings[key][1]
                count = 0

                print (vels(speed,turn))
                if (status == 14):
                    print (msg)
                status = (status + 1) % 15
            elif key == ' ' or key == 'k' :
                x = 0
                th = 0
                control_speed = 0
                control_turn = 0
            else:
                count = count + 1
                if count > 4:
                    x = 0
                    th = 0
                if (key == '\x03'):
                    break

            target_speed = speed * x
            target_turn = turn * th

            if target_speed > control_speed:
                control_speed = min( target_speed, control_speed + 0.02 )
            elif target_speed < control_speed:
                control_speed = max( target_speed, control_speed - 0.02 )
            else:
                control_speed = target_speed

            if target_turn > control_turn:
                control_turn = min( target_turn, control_turn + 0.1 )
            elif target_turn < control_turn:
                control_turn = max( target_turn, control_turn - 0.1 )
            else:
                control_turn = target_turn

            twist = Twist()
            twist.linear.x = control_speed; twist.linear.y = 0; twist.linear.z = 0
            twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = control_turn

            keyboard = Keyboard()
            keyboard.twist = twist
            keyboard.command = string
            pub.publish(keyboard)

            #print("loop: {0}".format(count))
            #print("target: vx: {0}, wz: {1}".format(target_speed, target_turn))
            #print("publihsed: vx: {0}, wz: {1}".format(twist.linear.x, twist.angular.z))

    except:
        print e

    finally:
        twist = Twist()
        twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
        pub.publish(twist)

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

