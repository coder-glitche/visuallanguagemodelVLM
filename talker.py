import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ActionPublisher(Node):
    def __init__(self):
        super().__init__('action_publisher')
        self.publisher_ = self.create_publisher(String, 'action_status', 10)
        self.get_logger().info('Action Publisher Node initialized. Type "done" to publish.')

    def publish_action_done(self):
        msg = String()
        msg.data = 'action done'
        self.publisher_.publish(msg)
        self.get_logger().info('Published: action done')

def main(args=None):
    rclpy.init(args=args)
    action_publisher = ActionPublisher()

    try:
        while rclpy.ok():
            user_input = input()
            if user_input.lower() == 'done':
                action_publisher.publish_action_done()
            elif user_input.lower() == 'exit':
                break
    except KeyboardInterrupt:
        pass
    finally:
        action_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
