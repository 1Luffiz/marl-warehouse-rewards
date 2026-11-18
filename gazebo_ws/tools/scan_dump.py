"""scan_dump.py – снимок лидаров трёх роботов в Gazebo для сравнения с быстрым симулятором."""
import json
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

N_BEAMS, RANGE = 16, 3.0

class ScanDump(Node):
    def __init__(self):
        super().__init__("scan_dump")
        self.got = {}
        for i in (1, 2, 3):
            self.create_subscription(LaserScan, f"/robot{i}/scan", lambda m, i=i: self.on_scan(i, m), 10)

    def on_scan(self, i, m):
        r = np.array(m.ranges, dtype=float)
        r[~np.isfinite(r)] = RANGE
        n = len(r)
        k = n // N_BEAMS
        w = min(2, k // 2)
        idx = (np.arange(N_BEAMS)[:, None] * k + np.arange(-w, w + 1)[None, :]) % n
        self.got[i] = np.clip(np.median(r[idx], axis=1), 0.0, RANGE).tolist()

def main():
    rclpy.init()
    node = ScanDump()
    t0 = time.time()
    while len(node.got) < 3 and time.time() - t0 < 20:
        rclpy.spin_once(node, timeout_sec=0.2)
    json.dump(node.got, open(sys.argv[1] if len(sys.argv) > 1 else "scan.json", "w"))
    print("получено сканов:", len(node.got))
    rclpy.shutdown()

if __name__ == "__main__":
    main()
