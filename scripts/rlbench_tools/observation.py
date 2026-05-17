from pyrep.const import RenderMode
from rlbench.observation_config import ObservationConfig


def _disable_camera(cam) -> None:
    cam.rgb = False
    cam.depth = False
    cam.point_cloud = False
    cam.masks = False


def _enable_rgb_camera(cam, width: int, height: int, renderer: str) -> None:
    cam.image_size = [width, height]
    cam.rgb = True
    cam.depth = False
    cam.point_cloud = False
    cam.masks = False
    cam.depth_in_meters = False
    cam.masks_as_one_channel = False
    cam.render_mode = RenderMode.OPENGL3 if renderer == "opengl3" else RenderMode.OPENGL


def build_obs_config(width: int, height: int, renderer: str) -> ObservationConfig:
    """
    Keep low-dimensional robot states, but only render right-shoulder RGB.
    This avoids saving unused camera tensors in LOW_DIM_PICKLE.
    """
    obs = ObservationConfig()
    obs.set_all(True)

    # Disable unused cameras.
    _disable_camera(obs.left_shoulder_camera)
    _disable_camera(obs.overhead_camera)
    _disable_camera(obs.wrist_camera)
    _disable_camera(obs.front_camera)

    # Enable only right shoulder RGB.
    _enable_rgb_camera(obs.right_shoulder_camera, width, height, renderer)

    obs.gripper_touch_forces = False
    return obs