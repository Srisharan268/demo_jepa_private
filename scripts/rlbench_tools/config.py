from dataclasses import dataclass
from typing import Tuple
import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CAMERA_JSON = os.path.join(CODE_DIR, "cams_extrinsics.json")

@dataclass
class RetargetConfig:
    save_path: str = "/tmp/paired_dataset/"
    task: str = ""
    source_robot: str = "panda"
    robots: Tuple[str, str] = ("panda", "sawyer")

    image_width: int = 640
    image_height: int = 480
    renderer: str = "opengl3"
    headless: bool = False

    variations: int = -1
    total_episodes: int = 300
    seed_master: int = 114514

    arm_max_velocity: float = 1.0
    arm_max_acceleration: float = 4.0
    static_positions: bool = False
    dt: float = 0.05

    max_demo_attempts: int = 10
    retries_per_pair: int = 2

    settle_pos_eps: float = 1e-3
    settle_ori_eps_deg: float = 2.0
    settle_max_steps: int = 40

    camera_json: str = DEFAULT_CAMERA_JSON