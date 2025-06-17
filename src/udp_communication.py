import socket
import json
import threading
import time
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
