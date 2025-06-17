#!/usr/bin/env python3
import rospy
from std_msgs.msg import String, Int32
import time

import sys

print(sys.path)


class RobotSimulator:
    """Simulates the robot's ROS communication behavior"""

    def __init__(self):
        rospy.init_node("robot_simulator", anonymous=True)

        # Subscribe to sweet selection requests
        self.selection_sub = rospy.Subscriber(
            "/sweet_selection", Int32, self.handle_selection
        )

        # Publisher for robot state updates
        self.state_pub = rospy.Publisher("/current_state", String, queue_size=10)

        rospy.loginfo("Robot Simulator started")

    def handle_selection(self, msg: Int32) -> None:
        """Simulates robot's response to a sweet selection request"""
        sweet_id = msg.data
        rospy.loginfo(f"Received selection request for sweet ID: {sweet_id}")

        # Simulate state transitions
        states = [
            "SEARCHING_FOR_SWEET",
            "EXECUTING_PICKUP",
            "OBJECT_GIVEN",
            "WAITING_FOR_SELECTION",
        ]

        for state in states:
            rospy.loginfo(f"Transitioning to state: {state}")
            self.state_pub.publish(state)
            time.sleep(2)  # Simulate processing time

    def run(self):
        """Start the robot simulator"""
        # Initial state
        self.state_pub.publish("WAITING_FOR_SELECTION")
        rospy.spin()


if __name__ == "__main__":
    try:
        simulator = RobotSimulator()
        simulator.run()
    except rospy.ROSInterruptException:
        pass
