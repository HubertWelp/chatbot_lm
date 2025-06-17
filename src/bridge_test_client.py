#!/usr/bin/env python3
import rospy

# from ros_bridge import ChatbotRosBridge, SweetType, RobotState
import time

import rospy
from std_msgs.msg import String, Int32
from enum import Enum
from typing import Optional, Callable
from dataclasses import dataclass


class SweetType(Enum):
    """Enum representing the available sweet types"""

    SNICKERS = 0
    MILKY_WAY = 1
    MAOAM = 2
    KINDERRIEGEL = 3


@dataclass
class RobotState:
    """Data class representing the robot's current state"""

    state: str
    timestamp: float


class ChatbotRosBridge:
    """Bridge class for handling ROS communication between chatbot and robot"""

    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("chatbot_ros_bridge", anonymous=False)

        # Publisher for sweet selection
        self.sweet_selection_pub = rospy.Publisher(
            "/sweet_selection", Int32, queue_size=10
        )

        # Subscriber for robot state
        self.state_sub = rospy.Subscriber("/current_state", String, self.state_callback)

        # Current robot state
        self.current_state: Optional[RobotState] = None

        # Callback for state changes
        self.state_change_callback: Optional[Callable[[RobotState], None]] = None

        # Wait for connections
        rospy.sleep(1)
        rospy.loginfo("ChatbotRosBridge initialized")

    def state_callback(self, msg: String) -> None:
        """Callback function for robot state updates"""
        self.current_state = RobotState(state=msg.data, timestamp=rospy.get_time())
        print("msg.data")
        if self.state_change_callback:
            self.state_change_callback(self.current_state)

    def register_state_callback(self, callback: Callable[[RobotState], None]) -> None:
        """Register a callback for state changes"""
        self.state_change_callback = callback

    def request_sweet(self, sweet_type: SweetType) -> bool:
        """
        Request the robot to retrieve a specific sweet

        Args:
            sweet_type: The type of sweet to retrieve

        Returns:
            bool: True if request was sent successfully
        """
        try:
            msg = Int32()
            msg.data = sweet_type.value
            self.sweet_selection_pub.publish(msg)
            return True
        except rospy.ROSException:
            rospy.logerr("Failed to publish sweet selection request")
            return False

    def get_current_state(self) -> Optional[RobotState]:
        """Get the current state of the robot"""
        return self.current_state

    def wait_for_state(self, target_state: str, timeout: float = 30.0) -> bool:
        """
        Wait for the robot to reach a specific state

        Args:
            target_state: The state to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if target state was reached, False if timeout occurred
        """
        start_time = rospy.get_time()
        rate = rospy.Rate(10)  # 10Hz

        while not rospy.is_shutdown():
            if self.current_state and self.current_state.state == target_state:
                return True

            if rospy.get_time() - start_time > timeout:
                return False
            rospy.sleep(1)
            rate.sleep()

    def spin(self) -> None:
        """Start the ROS event loop"""
        rospy.spin()


class BridgeTester:
    """Tests the ChatbotRosBridge functionality"""

    def __init__(self):
        self.bridge = ChatbotRosBridge()
        self.bridge.register_state_callback(self.state_change_handler)

    def state_change_handler(self, state: RobotState) -> None:
        """Handles state change notifications from the bridge"""
        rospy.loginfo(f"State changed to: {state.state} at {state.timestamp}")

    def run_tests(self):
        """Executes a series of tests"""
        rospy.loginfo("Starting bridge tests...")

        # Test each sweet type
        for sweet in SweetType:
            rospy.loginfo(f"\nTesting selection of {sweet.name}")

            # Request the sweet
            success = self.bridge.request_sweet(sweet)
            if not success:
                rospy.logerr(f"Failed to request {sweet.name}")
                continue

            # Wait for completion
            if self.bridge.wait_for_state("WAITING_FOR_SELECTION", timeout=60.0):
                rospy.loginfo(f"Successfully completed {sweet.name} request")
            else:
                rospy.logwarn(f"Timeout waiting for {sweet.name} request completion")

            # Wait between tests
            time.sleep(2)

        rospy.loginfo("Test sequence completed")


def main():
    try:
        tester = BridgeTester()
        tester.run_tests()

        # Keep node running to receive messages
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
