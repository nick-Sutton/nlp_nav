#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from ollama import chat


COORD_DICT = {
    "living room": (-4.0,  4.0),
    "bedroom":     (-4.5, -0.9),
    "kitchen":     (-0.8,  2.00),
    "laundry":     (2.8,  2.00),
    "office":      ( 7.0,  2.0),
    "library":     ( 8.2, -0.5),
}

SYSTEM_PROMPT = (
    'You are a robot navigation assistant. Interpret the user request and respond '
    'with exactly one room name from this list (spelling must match exactly): '
    f'{list(COORD_DICT.keys())}. Output only the room name, nothing else.'
)


class LanguageProcessor(Node):

    def __init__(self):
        super().__init__('language_processor')
        self.publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)

    def run(self):
        while rclpy.ok():
            print("\nWhat would you like me to do? (Type 'quit' to stop)")
            try:
                prompt = input('> ').strip()
            except EOFError:
                break

            if prompt.lower() == 'quit':
                break
            if not prompt:
                continue

            print(f"Thinking...")

            room = None
            for attempt in range(5):
                response = chat(
                    model='qwen2.5:3b',
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user',   'content': prompt},
                    ],
                )
                candidate = response.message.content.strip().lower()
                if candidate in COORD_DICT:
                    room = candidate
                    break
                self.get_logger().debug(f'Attempt {attempt+1}: unrecognised response "{candidate}"')

            if room is None:
                print("Sorry, I couldn't identify a room from that. Please try again.")
                continue

            x, y = COORD_DICT[room]
            print(f"Navigating to {room} ({x}, {y})")

            msg = PoseStamped()
            msg.header.frame_id = 'map'
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.position.x = float(x)
            msg.pose.position.y = float(y)
            msg.pose.orientation.w = 1.0
            self.publisher.publish(msg)


def main():
    rclpy.init()
    node = LanguageProcessor()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
