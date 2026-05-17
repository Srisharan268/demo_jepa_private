from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.backend.utils import task_file_to_task_class

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete


def make_env(
    robot_setup: str,
    obs_config: ObservationConfig,
    arm_mode,
    headless: bool,
    arm_max_velocity: float,
    arm_max_acceleration: float,
    static_positions: bool,
    dt: float,
) -> Environment:
    """
    Create and launch an RLBench environment.

    arm_mode can be:
        - JointVelocity()
        - EndEffectorPoseViaIK()
        - other RLBench arm action modes
    """
    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=arm_mode,
            gripper_action_mode=Discrete(),
        ),
        obs_config=obs_config,
        headless=headless,
        arm_max_velocity=arm_max_velocity,
        arm_max_acceleration=arm_max_acceleration,
        robot_setup=robot_setup,
        static_positions=static_positions,
    )

    env.launch()

    # RLBench没有公开接口设置这些，所以这里沿用私有属性。
    env._pyrep.set_simulation_timestep(dt)
    env._robot.arm.set_control_loop_enabled(True)

    return env


def get_task_env(env: Environment, task_name: str):
    """
    Get task environment from RLBench task file name.

    Example:
        task_name = "close_box"
    """
    task_class = task_file_to_task_class(task_name)
    return env.get_task(task_class)


def make_minimal_env(
    robot_setup: str,
    headless: bool,
    static_positions: bool,
    dt: float = 0.05,
) -> Environment:
    """
    Create a lightweight RLBench environment for metadata queries.

    Used for:
        - variation_count()
        - checking whether a task can be loaded

    It still needs a valid action mode, so we use JointVelocity().
    """
    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(),
            gripper_action_mode=Discrete(),
        ),
        obs_config=ObservationConfig(),
        headless=headless,
        robot_setup=robot_setup,
        static_positions=static_positions,
    )

    env.launch()

    env._pyrep.set_simulation_timestep(dt)
    env._robot.arm.set_control_loop_enabled(True)

    return env


def get_variation_count(
    task_name: str,
    robot_setup: str,
    headless: bool,
    static_positions: bool,
    dt: float = 0.05,
) -> int:
    """
    Query the number of variations for a given RLBench task.
    """
    env = make_minimal_env(
        robot_setup=robot_setup,
        headless=headless,
        static_positions=static_positions,
        dt=dt,
    )

    try:
        task_env = get_task_env(env, task_name)
        return task_env.variation_count()
    finally:
        env.shutdown()