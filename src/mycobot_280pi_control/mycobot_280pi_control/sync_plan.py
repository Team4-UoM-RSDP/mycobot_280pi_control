import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time
import math
import pymycobot
from packaging import version

# min low version require
MIN_REQUIRE_VERSION = "3.6.1"

current_verison = pymycobot.__version__
print("current pymycobot library version: {}".format(current_verison))
if version.parse(current_verison) < version.parse(MIN_REQUIRE_VERSION):
    raise RuntimeError(
        "The version of pymycobot library must be greater than {} or higher. The current version is {}. Please upgrade the library version.".format(
            MIN_REQUIRE_VERSION, current_verison
        )
    )
else:
    print("pymycobot library version meets the requirements!")
    from pymycobot import MyCobot280


class Slider_Subscriber(Node):
    def __init__(self):
        super().__init__("control_sync_plan")

        # Declare parameters
        self.declare_parameter("port", "/dev/ttyAMA0")
        self.declare_parameter("baud", 1000000)
        self.declare_parameter("joint_topic", "joint_states")
        self.declare_parameter("speed", 35)
        self.declare_parameter(
            "sync_rate_hz", 0.0
        )  # 0 = sync every message, >0 = limit rate

        # Get parameters
        port = self.get_parameter("port").get_parameter_value().string_value
        baud = self.get_parameter("baud").get_parameter_value().integer_value
        joint_topic = (
            self.get_parameter("joint_topic").get_parameter_value().string_value
        )
        self.speed = self.get_parameter("speed").get_parameter_value().integer_value
        sync_rate = (
            self.get_parameter("sync_rate_hz").get_parameter_value().double_value
        )

        self.get_logger().info("port:%s, baud:%d" % (port, baud))
        self.get_logger().info("Subscribing to topic: %s" % joint_topic)
        self.get_logger().info("Movement speed: %d" % self.speed)

        # Initialize hardware connection
        self.mc = MyCobot280(port, baud)
        time.sleep(0.05)
        if self.mc.get_fresh_mode() == 0:
            self.mc.set_fresh_mode(1)
            time.sleep(0.05)
        self.get_logger().info("MyCobot280 hardware connection established")

        # Joint name mapping - update these to match your URDF
        self.arm_joint_names = [
            "link1_to_link2",
            "link2_to_link3",
            "link3_to_link4",
            "link4_to_link5",
            "link5_to_link6",
            "link6_to_link6_flange",
        ]
        self.joint_limits_deg = self._load_joint_limits_deg()
        self.gripper_joint_name = "gripper_controller"
        # Range comes from adaptive_gripper.urdf.xacro limits (-0.7, 0.15)
        self.gripper_lower_limit = -0.7
        self.gripper_upper_limit = 0.15
        self.gripper_value_min = 0
        self.gripper_value_max = 100
        self.gripper_type = 1  # 1 = Adaptive gripper, 3 = Parallel gripper
        self.last_gripper_value = None
        self.gripper_supported = hasattr(self.mc, "set_gripper_value")
        if not self.gripper_supported:
            self.get_logger().warn(
                "Current pymycobot driver does not expose set_gripper_value(); gripper commands disabled."
            )

        # Rate limiting
        self.last_sync_time = time.time()
        self.min_sync_interval = 1.0 / sync_rate if sync_rate > 0 else 0.0

        # Subscribe to joint states
        self.subscription = self.create_subscription(
            JointState, joint_topic, self.listener_callback, 10
        )
        self.subscription

    def listener_callback(self, msg):
        # Rate limiting check
        current_time = time.time()
        if self.min_sync_interval > 0:
            if current_time - self.last_sync_time < self.min_sync_interval:
                return  # Skip this message
            self.last_sync_time = current_time

        # Create dictionary mapping joint names to positions
        joint_state_dict = {name: msg.position[i] for i, name in enumerate(msg.name)}

        # Extract angles in correct order
        data_list = []
        missing_joints = []
        for joint in self.arm_joint_names:
            if joint in joint_state_dict:
                radians_to_angles = round(math.degrees(joint_state_dict[joint]), 3)
                data_list.append(radians_to_angles)
            else:
                missing_joints.append(joint)

        # Only send if we have all 6 joints
        if len(data_list) == 6:
            clamped_angles, had_clamp = self._clamp_angles_deg(data_list)
            if had_clamp:
                self.get_logger().warn(
                    "Clamped joint angles to limits. Command: {} -> {}".format(
                        data_list, clamped_angles
                    ),
                    throttle_duration_sec=2.0,
                )
            self.get_logger().debug("Sending angles: {}".format(clamped_angles))
            try:
                self.mc.send_angles(clamped_angles, self.speed)
            except Exception as e:
                self.get_logger().error("Failed to send angles to robot: {}".format(e))
        elif missing_joints:
            self.get_logger().warn(
                "Missing joints in message: {}. Available: {}".format(
                    missing_joints, list(joint_state_dict.keys())
                ),
                throttle_duration_sec=5.0,
            )

        if self.gripper_joint_name in joint_state_dict:
            self._sync_gripper(joint_state_dict[self.gripper_joint_name])
        elif self.gripper_supported:
            self.get_logger().debug(
                "JointState missing gripper joint '{}'".format(self.gripper_joint_name)
            )

    def _sync_gripper(self, gripper_angle_rad):
        if not self.gripper_supported:
            return

        gripper_value = self._gripper_angle_to_value(gripper_angle_rad)
        if gripper_value == self.last_gripper_value:
            return

        try:
            # Ensure speed is within 0-100 range for gripper
            gripper_speed = max(0, min(100, self.speed))
            # Set gripper with type parameter (1 = Adaptive gripper)
            self.mc.set_gripper_value(gripper_value, gripper_speed, self.gripper_type)
            self.last_gripper_value = gripper_value
            self.get_logger().debug(
                "Sending gripper value {} (speed: {}) for {:.3f} rad".format(
                    gripper_value, gripper_speed, gripper_angle_rad
                )
            )
        except Exception as e:
            self.get_logger().error(
                "Failed to send gripper command: {}".format(e),
                throttle_duration_sec=2.0,
            )

    def _gripper_angle_to_value(self, angle_rad):
        clamped = max(
            self.gripper_lower_limit, min(self.gripper_upper_limit, angle_rad)
        )
        normalized = (clamped - self.gripper_lower_limit) / (
            self.gripper_upper_limit - self.gripper_lower_limit
        )
        value = int(
            round(
                self.gripper_value_min
                + normalized * (self.gripper_value_max - self.gripper_value_min)
            )
        )
        return value

    def _load_joint_limits_deg(self):
        limits = []
        try:
            for joint_index in range(1, 7):
                joint_max = self.mc.get_joint_max_angle(joint_index)
                time.sleep(0.05)
                joint_min = self.mc.get_joint_min_angle(joint_index)
                time.sleep(0.05)
                limits.append((joint_min, joint_max))
            self.get_logger().info("Loaded joint limits (deg): {}".format(limits))
        except Exception as e:
            self.get_logger().warn(
                "Failed to read joint limits from hardware: {}. Using no limits.".format(
                    e
                )
            )
            limits = [(float("-inf"), float("inf"))] * 6
        return limits

    def _clamp_angles_deg(self, angles_deg):
        clamped = []
        had_clamp = False
        for i, angle in enumerate(angles_deg):
            min_lim, max_lim = self.joint_limits_deg[i]
            if angle < min_lim:
                clamped.append(min_lim)
                had_clamp = True
            elif angle > max_lim:
                clamped.append(max_lim)
                had_clamp = True
            else:
                clamped.append(angle)
        return clamped, had_clamp


def main(args=None):
    try:
        rclpy.init(args=args)
        slider_subscriber = Slider_Subscriber()
        rclpy.spin(slider_subscriber)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
