import os
from glob import glob

from setuptools import setup

package_name = "marl_gazebo"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.sdf")),
        (os.path.join("share", package_name, "models"), glob("models/*")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "config", "layouts"), glob("config/layouts/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Volov G.A.",
    maintainer_email="1Luffiz@users.noreply.github.com",
    description="Gazebo-стенд для мультиагентной навигации группы роботов (MARL)",
    license="MIT",
    entry_points={
        "console_scripts": [
            "policy_node = marl_gazebo.policy_node:main",
        ],
    },
)
