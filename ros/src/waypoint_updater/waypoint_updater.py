#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial import KDTree
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number
MAX_DECEL = 1.0


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater', log_level=rospy.INFO)

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_wp_index = None

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)  # 50 Hz; 30 Hz may work as well
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        closest_coord = self.waypoints_2d[closest_idx]

        if closest_idx == 0:
            # Check if closest is ahead or behind vehicle
            prev_coord = self.waypoints_2d[closest_idx - 1]

            # Equation for hyperplane through closest_coords
            cl_vect = np.array(closest_coord)
            prev_vect = np.array(prev_coord)
            pos_vect = np.array([x, y])

            val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)
            if val > 0:
                closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx

    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose

            # Two waypoints back from the line so front of the car stops at line
            stop_idx = max(self.stopline_wp_index - closest_idx - 2, 0)

            # Decelerate linearly
            # velocity = dist / 5

            # Decelerate by S-curve
            # velocity = wp.twist.twist.linear.x * (1. - 1. / (1. + math.exp(-(i - stop_idx + 30.) / 6.)))

            dist = self.distance(waypoints, i, stop_idx)
            velocity = math.sqrt(2 * MAX_DECEL * dist)
            if velocity < 1.:
                velocity = 0.

            p.twist.twist.linear.x = min(velocity, wp.twist.twist.linear.x)
            temp.append(p)

        return temp

    def generate_lane(self, closest_idx):
        msg = Lane()
        msg.header = self.base_waypoints.header
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]

        if self.stopline_wp_index is not None and 0 <= self.stopline_wp_index <= farthest_idx:
            msg.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
        else:
            msg.waypoints = base_waypoints

        return msg

    def publish_waypoints(self, closest_idx):
        # NOTE: the first waypoint in the list published to /final_waypoints should be
        # the first waypoint that is currently ahead of the car.
        msg = self.generate_lane(closest_idx)

        s = " ".join(["{:.1f}".format(wp.twist.twist.linear.x) for wp in msg.waypoints[:50]])
        rospy.loginfo("{}: {}".format(closest_idx, s))

        self.final_waypoints_pub.publish(msg)

    def pose_cb(self, msg):
        self.pose = msg

    @staticmethod
    def _get_waypoint_xy(waypoint):
        return waypoint.pose.pose.position.x, waypoint.pose.pose.position.y

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [self._get_waypoint_xy(wp) for wp in self.base_waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.stopline_wp_index = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    @staticmethod
    def get_waypoint_velocity(waypoint):
        return waypoint.twist.twist.linear.x

    @staticmethod
    def set_waypoint_velocity(waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    @staticmethod
    def distance(waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
