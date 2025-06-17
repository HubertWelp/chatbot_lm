#! /home/student/catkin_ws/src/chatbot_lm/.venv python
import rospy
from std_msgs.msg import String, Int32
from typing import Optional, Callable
from dataclasses import dataclass

# from sweet_types import SweetType
# from udp_communication import UDPCommunication, MessageType

import socket
import json
import threading
import time
import argparse
from dataclasses import dataclass
from typing import Callable, Optional, Dict
from enum import Enum, auto


class MessageType(Enum):
    """Defines the types of messages that can be exchanged"""

    SWEET_SELECTION = auto()
    ROBOT_STATUS = auto()
    HEARTBEAT = auto()


@dataclass
class Message:
    """Data structure for messages exchanged between nodes"""

    type: MessageType
    data: any
    timestamp: float


class UDPCommunication:
    """Handles bidirectional UDP communication between nodes"""

    def __init__(
        self,
        local_ip: str,
        local_port: int,
        remote_ip: str,
        remote_port: int,
        buffer_size: int = 4096,
    ):
        """Initialize UDP communication

        Args:
            local_ip (str): IP address to bind to locally
            local_port (int): Port to bind to locally
            remote_ip (str): IP address of remote endpoint
            remote_port (int): Port of remote endpoint
            buffer_size (int): Size of the receive buffer in bytes
        """
        self.local_addr = (local_ip, local_port)
        self.remote_addr = (remote_ip, remote_port)
        self.buffer_size = buffer_size

        try:
            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(self.local_addr)

            # Set socket timeout to prevent blocking indefinitely
            self.socket.settimeout(1.0)

            print(
                f"UDP Communication initialized - Local: {local_ip}:{local_port}, Remote: {remote_ip}:{remote_port}"
            )
        except Exception as e:
            print(f"Failed to initialize UDP socket: {e}")
            raise

        # Message handlers for different message types
        self.handlers: Dict[MessageType, Callable[[Message], None]] = {}

        # Start listener thread
        self.running = True
        self.listener_thread = threading.Thread(target=self._listen)
        self.listener_thread.daemon = True
        self.listener_thread.start()

    def register_handler(
        self, msg_type: MessageType, handler: Callable[[Message], None]
    ):
        """Register a callback handler for a specific message type

        Args:
            msg_type (MessageType): Type of message to handle
            handler (Callable[[Message], None]): Callback function to handle the message
        """
        self.handlers[msg_type] = handler
        print(f"Registered handler for message type: {msg_type.name}")

    def send_message(self, msg_type: MessageType, data: any) -> bool:
        """Send a message to the remote endpoint

        Args:
            msg_type (MessageType): Type of message to send
            data: Data to send in the message

        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        if not self.running:
            print("Cannot send message: UDP Communication is not running")
            return False

        try:
            message = Message(type=msg_type, data=data, timestamp=time.time())

            # Convert message to JSON
            json_data = json.dumps(
                {
                    "type": message.type.name,
                    "data": message.data,
                    "timestamp": message.timestamp,
                }
            )

            # Send message
            bytes_sent = self.socket.sendto(json_data.encode(), self.remote_addr)

            # Verify message was sent
            if bytes_sent > 0:
                print(f"Sent {msg_type.name} message ({bytes_sent} bytes)")
                return True
            return False

        except Exception as e:
            print(f"Error sending message: {e}")
            return False

    def _listen(self):
        """Listen for incoming messages and dispatch to appropriate handlers"""
        while self.running:
            try:
                # Receive data
                data, addr = self.socket.recvfrom(self.buffer_size)

                # Parse JSON message
                json_data = json.loads(data.decode())

                # Create message object
                message = Message(
                    type=MessageType[json_data["type"]],
                    data=json_data["data"],
                    timestamp=json_data["timestamp"],
                )

                # Dispatch to handler if one exists
                if message.type in self.handlers:
                    try:
                        self.handlers[message.type](message)
                    except Exception as e:
                        print(f"Error in message handler: {e}")

            except socket.timeout:
                # This is expected, continue listening
                continue
            except json.JSONDecodeError as e:
                print(f"Error decoding message: {e}")
            except Exception as e:
                if (
                    self.running
                ):  # Only print errors if we're still supposed to be running
                    print(f"Error in listener thread: {e}")

    def close(self):
        """Close the UDP connection and stop the listener thread"""
        print("Closing UDP Communication...")
        self.running = False
        try:
            self.socket.close()
            print("UDP socket closed successfully")
        except Exception as e:
            print(f"Error closing UDP socket: {e}")


class SweetType(Enum):
    SNICKERS = 0
    MILKYWAY = 1
    MAOAM = 2
    KINDERRIEGEL = 3

    @classmethod
    def from_string(cls, text: str) -> Optional["SweetType"]:
        """Convert a string to a SweetType enum value."""
        normalized = text.lower().replace(" ", "")
        mapping = {
            "snickers": cls.SNICKERS,
            "milkyway": cls.MILKYWAY,
            "milky-way": cls.MILKYWAY,
            "milky_way": cls.MILKYWAY,
            "maoam": cls.MAOAM,
            "kinderriegel": cls.KINDERRIEGEL,
            "kinder-riegel": cls.KINDERRIEGEL,
            "kinder_riegel": cls.KINDERRIEGEL,
        }
        return mapping.get(normalized)

    @classmethod
    def to_display_name(cls, sweet_type: "SweetType") -> str:
        """Convert a SweetType enum value to its display name."""
        display_names = {
            cls.SNICKERS: "Snickers",
            cls.MILKYWAY: "Milky Way",
            cls.MAOAM: "Maoam",
            cls.KINDERRIEGEL: "Kinderriegel",
        }
        return display_names[sweet_type]

    @property
    def display_name(self) -> str:
        """Get the display name for this sweet type."""
        return self.to_display_name(self)

    @classmethod
    def get_all_variants(cls) -> Dict[str, "SweetType"]:
        """Get a dictionary of all possible string variants mapping to their SweetType."""
        variants = {
            "snickers": cls.SNICKERS,
            "milkyway": cls.MILKYWAY,
            "milky way": cls.MILKYWAY,
            "milky-way": cls.MILKYWAY,
            "milky_way": cls.MILKYWAY,
            "maoam": cls.MAOAM,
            "kinderriegel": cls.KINDERRIEGEL,
            "kinder riegel": cls.KINDERRIEGEL,
            "kinder-riegel": cls.KINDERRIEGEL,
            "kinder_riegel": cls.KINDERRIEGEL,
        }
        return variants


def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close
    #   ip = "172.20.11.26"
    return ip


@dataclass
class RobotState:
    """Data class representing the robot's current state"""

    state: str
    timestamp: float


class ChatbotRosBridge:
    """Bridge class for handling ROS communication between chatbot and robot"""

    def __init__(self, remote_ip_adress: str = "127.0.0.1"):
        # Initialize the ROS node
        rospy.init_node("chatbot_ros_bridge", anonymous=True)

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

        if remote_ip_adress == "127.0.0.1":
            local_ip_adress: str = "127.0.0.1"
        else:
            local_ip_adress: str = get_local_ip()
        print(f"remote IP-Adress: {remote_ip_adress} ")
        print(f"local IP-Adress: {local_ip_adress} ")
        self.udp = UDPCommunication(
            local_ip=local_ip_adress,
            local_port=5000,
            remote_ip=remote_ip_adress,
            remote_port=5001,
        )

        # Register handlers for incoming messages
        self.udp.register_handler(
            MessageType.SWEET_SELECTION, self._handle_sweet_selection
        )

    def _handle_sweet_selection(self, message):
        # Convert UDP message to ROS message and publish
        msg = Int32()
        msg.data = message.data
        self.sweet_selection_pub.publish(msg)

    def state_callback(self, msg: String) -> None:
        """
        Callback function for robot state updates.

        This method processes incoming robot state messages, updates the internal state,
        forwards the state via UDP to the chatbot orchestrator, and triggers any registered
        state change callbacks.

        Args:
            msg (String): ROS String message containing the robot state

        Note:
            The method handles potential errors during UDP transmission and callback execution
            to ensure system stability.
        """
        try:
            # Create new RobotState instance with current timestamp
            new_state = RobotState(
                state=msg.data.strip(),
                timestamp=rospy.get_time(),
            )

            # Update internal state
            self.current_state = new_state

            # Attempt to send state via UDP
            success = self.udp.send_message(
                MessageType.ROBOT_STATUS,
                {"state": new_state.state, "timestamp": new_state.timestamp},
            )

            if not success:
                rospy.logwarn("Failed to send robot state update via UDP")

            # Execute state change callback if registered
            if self.state_change_callback:
                try:
                    self.state_change_callback(new_state)
                except Exception as callback_error:
                    rospy.logerr(f"Error in state change callback: {callback_error}")

            # Log state change for debugging
            rospy.logdebug(f"Robot state updated to: {new_state.state}")

        except AttributeError as e:
            rospy.logerr(f"Invalid message format: {e}")
        except Exception as e:
            rospy.logerr(f"Unexpected error in state callback: {e}")

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
            rospy.loginfo(
                f"Requesting sweet: {sweet_type.display_name} (ID: {sweet_type.value})"
            )
            return True
        except rospy.ROSException:
            rospy.logerr(
                f"Failed to publish sweet selection request for {sweet_type.display_name}"
            )
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

            rate.sleep()

    def spin(self) -> None:
        """Start the ROS event loop"""
        rospy.spin()


def state_change_handler(state: RobotState) -> None:
    """Example callback for state changes"""
    rospy.loginfo(f"Robot state changed to: {state.state}")


if __name__ == "__main__":

    print("Hello ROS-Bridge")
    parser = argparse.ArgumentParser(description="IP-Adress parser")
    parser.add_argument("--IP", help="IP-Adress of chatbot_orchestrator")

    args = parser.parse_args()
    if args.IP:
        remote_ip: str = args.IP
    else:
        remote_ip: str = "127.0.0.1"

    try:
        bridge = ChatbotRosBridge(remote_ip)
        bridge.register_state_callback(state_change_handler)

        # # Example sweet request
        # bridge.request_sweet(SweetType.SNICKERS)

        # Wait for robot to finish
        if bridge.wait_for_state("WAITING_FOR_SELECTION"):
            rospy.loginfo("Robot is ready for next selection")

        bridge.spin()

    except rospy.ROSInterruptException:
        pass
