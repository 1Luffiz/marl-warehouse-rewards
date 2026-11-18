"""motion_measure.py – замер движения робота в Gazebo."""
import json
import math
import sys

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

ACTS = [("вперёд быстро", 0.40, 0.0), ("вперёд медленно", 0.15, 0.0),
        ("плавный поворот влево", 0.20, 1.0), ("плавный поворот вправо", 0.20, -1.0),
        ("разворот влево", 0.0, 1.5), ("разворот вправо", 0.0, -1.5)]

class KinTest(Node):
    def __init__(self, robot="robot2"):
        super().__init__("motion_measure")
        self.pose = None   # (x, y, yaw, t)
        self.create_subscription(Odometry, f"/{robot}/odom", self.on_odom, 10)
        self.pub = self.create_publisher(Twist, f"/{robot}/cmd_vel", 10)

    def on_odom(self, m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw, t)

    def drive(self, tw, until):
        while self.pose[3] < until:
            self.pub.publish(tw)
            rclpy.spin_once(self, timeout_sec=0.05)

    def run(self, v, w, dur=1.0, warmup=0.5):
        tw = Twist()
        tw.linear.x, tw.angular.z = v, w
        while self.pose is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.drive(tw, self.pose[3] + warmup)
        p0 = self.pose
        yaw, prev = 0.0, p0[2]
        while self.pose[3] - p0[3] < dur:
            self.pub.publish(tw)
            rclpy.spin_once(self, timeout_sec=0.05)
            yaw += math.atan2(math.sin(self.pose[2] - prev), math.cos(self.pose[2] - prev))
            prev = self.pose[2]
        p1 = self.pose
        dt = p1[3] - p0[3]
        return math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / dt, yaw / dt

def ru(x, fmt):
    return format(x, fmt).replace(".", ",").replace("-", "−")

def main():
    rclpy.init()
    node = KinTest()
    out = []
    for name, v, w in ACTS:
        mv, mw = node.run(v, w)
        print(f"{name:16s} команда v={v:.2f} w={w:+.1f}  измерено v={mv:.3f} w={mw:+.3f}")
        out.append(dict(name=name, v=ru(v, ".2f"), w=ru(w, ".1f"), mv=ru(mv, ".2f"), mw=ru(mw, ".2f")))
    node.run(0.0, 0.0, 0.3)   # остановить робота
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "motion.json", "w"), ensure_ascii=False, indent=1)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
