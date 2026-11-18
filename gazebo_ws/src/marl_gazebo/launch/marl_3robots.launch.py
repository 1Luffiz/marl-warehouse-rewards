"""marl_3robots.launch.py – запуск склада в Gazebo с тремя роботами."""

import os
from string import Template

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

COLORS = [(0.12, 0.47, 0.71), (0.84, 0.15, 0.16), (0.17, 0.63, 0.17)]

GOAL_SDF = """<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="disk">
        <pose>0 0 0.005 0 0 0</pose>
        <geometry><cylinder><radius>0.3</radius><length>0.01</length></cylinder></geometry>
        <material><ambient>0.1 0.8 0.1 1</ambient><diffuse>0.1 0.8 0.1 1</diffuse></material>
        <transparency>0.4</transparency>
      </visual>
      <visual name="pole">
        <pose>0 0 0.03 0 0 0</pose>
        <geometry><cylinder><radius>0.02</radius><length>0.06</length></cylinder></geometry>
        <material><ambient>0 0.4 0 1</ambient><diffuse>0 0.4 0 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>
"""

def launch_setup(context, *args, **kwargs):
    pkg = get_package_share_directory("marl_gazebo")
    world_name = LaunchConfiguration("world").perform(context)
    world = os.path.join(pkg, "worlds", f"{world_name}.sdf")
    bridge_cfg = os.path.join(pkg, "config", "bridge.yaml")
    scen_file = os.path.join(pkg, "config", f"scenarios_{world_name}.yaml")
    layout_file = os.path.join(pkg, "config", "layouts", f"{world_name}.yaml")
    with open(os.path.join(pkg, "models", "diff_robot.sdf.template")) as f:
        robot_tpl = Template(f.read())

    scenario_id = int(LaunchConfiguration("scenario").perform(context))
    headless = LaunchConfiguration("headless").perform(context).lower() == "true"
    use_policy = LaunchConfiguration("policy").perform(context).lower() == "true"
    model_path = LaunchConfiguration("model").perform(context)

    with open(scen_file) as f:
        scen = yaml.safe_load(f)["scenarios"][scenario_id]

    gz_args = f"-r -v 2 {world}"
    if headless:
        gz_args = f"-s -r -v 2 --headless-rendering {world}"

    actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")),
            launch_arguments={"gz_args": gz_args, "on_exit_shutdown": "true"}.items(),
        ),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="marl_bridge",
            parameters=[{"config_file": bridge_cfg, "use_sim_time": True}],
            output="screen",
        ),
    ]

    # Роботы
    for i, (x, y, yaw) in enumerate(scen["robots"]):
        name = f"robot{i + 1}"
        flag = 1 << (i + 1)
        sdf = robot_tpl.substitute(
            name=name, r=COLORS[i][0], g=COLORS[i][1], b=COLORS[i][2],
            flag=flag, mask=0x0FFFFFFF ^ flag)
        actions.append(Node(
            package="ros_gz_sim", executable="create", output="screen",
            arguments=["-world", world_name, "-name", name, "-string", sdf,
                       "-x", str(x), "-y", str(y), "-z", "0.0", "-Y", str(yaw)]))

    # Маркеры заказов на станциях (только визуальные, без столкновений)
    for j, (gx, gy) in enumerate(scen["goals"]):
        name = f"goal{j + 1}"
        actions.append(Node(
            package="ros_gz_sim", executable="create", output="screen",
            arguments=["-world", world_name, "-name", name,
                       "-string", GOAL_SDF.format(name=name),
                       "-x", str(gx), "-y", str(gy), "-z", "0.0"]))

    # Узел с обученной моделью – стартует после появления роботов
    if use_policy:
        actions.append(TimerAction(period=8.0, actions=[Node(
            package="marl_gazebo", executable="policy_node", name="marl_policy",
            output="screen",
            parameters=[{"model_path": model_path,
                         "scenarios_file": scen_file,
                         "layout_file": layout_file,
                         "deterministic": LaunchConfiguration("deterministic").perform(context).lower() == "true",
                         "scenario": scenario_id,
                         "use_sim_time": True}])]))
    return actions

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="warehouse", description="мир: warehouse или arena"),
        DeclareLaunchArgument("scenario", default_value="0", description="номер сценария из scenarios_<мир>.yaml"),
        DeclareLaunchArgument("headless", default_value="false", description="без графического окна"),
        DeclareLaunchArgument("policy", default_value="false", description="запустить обученную модель"),
        DeclareLaunchArgument("model", default_value="", description="путь к actor.npz (или папке прогона)"),
        DeclareLaunchArgument("deterministic", default_value="false", description="выбор наиболее вероятного действия"),
        OpaqueFunction(function=launch_setup),
    ])
