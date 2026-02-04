import time
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

mc = MyCobot280("/dev/ttyAMA0", 1000000)

angle_max = []
angle_min = []

for i in range(1, 7):
    joint_max = mc.get_joint_max_angle(i)
    angle_max.append(joint_max)
    time.sleep(0.05)
    joint_min = mc.get_joint_min_angle(i)
    angle_min.append(joint_min)

print("Max angle:", angle_max)
print("Min angle:", angle_min)


def main(args=None):
    try:
        pass
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
