import argparse

from config import RetargetConfig, DEFAULT_CAMERA_JSON
from retarget import run_collection


def parse_args() -> RetargetConfig:
    parser = argparse.ArgumentParser(
        description="Retarget RLBench demos with identical scene seeds."
    )

    parser.add_argument("--save_path", type=str, default="/tmp/paired_dataset/")
    parser.add_argument("--task", type=str, required=True)

    parser.add_argument("--source_robot", type=str, default="panda")
    parser.add_argument("--robots", nargs=2, default=["panda", "sawyer"])

    parser.add_argument("--image_size", nargs=2, type=int, default=[640, 480])
    parser.add_argument("--renderer", choices=["opengl", "opengl3"], default="opengl3")
    parser.add_argument("--headless", action="store_true")

    parser.add_argument("--variations", type=int, default=-1)
    parser.add_argument("--total_episodes", type=int, default=300)
    parser.add_argument("--seed_master", type=int, default=233333)

    parser.add_argument("--arm_max_velocity", type=float, default=1.0)
    parser.add_argument("--arm_max_acceleration", type=float, default=4.0)
    parser.add_argument("--static_positions", action="store_true")
    parser.add_argument("--dt", type=float, default=0.05)

    parser.add_argument("--max_demo_attempts", type=int, default=10)
    parser.add_argument("--retries_per_pair", type=int, default=2)

    parser.add_argument("--settle_pos_eps", type=float, default=1e-3)
    parser.add_argument("--settle_ori_eps_deg", type=float, default=2.0)
    parser.add_argument("--settle_max_steps", type=int, default=40)

    parser.add_argument(
        "--camera_json",
        type=str,
        default=DEFAULT_CAMERA_JSON,
    )

    args = parser.parse_args()

    return RetargetConfig(
        save_path=args.save_path,
        task=args.task,
        source_robot=args.source_robot,
        robots=tuple(args.robots),

        image_width=args.image_size[0],
        image_height=args.image_size[1],
        renderer=args.renderer,
        headless=args.headless,

        variations=args.variations,
        total_episodes=args.total_episodes,
        seed_master=args.seed_master,

        arm_max_velocity=args.arm_max_velocity,
        arm_max_acceleration=args.arm_max_acceleration,
        static_positions=args.static_positions,
        dt=args.dt,

        max_demo_attempts=args.max_demo_attempts,
        retries_per_pair=args.retries_per_pair,

        settle_pos_eps=args.settle_pos_eps,
        settle_ori_eps_deg=args.settle_ori_eps_deg,
        settle_max_steps=args.settle_max_steps,

        camera_json=args.camera_json,
    )


def main() -> None:
    cfg = parse_args()
    run_collection(cfg)


if __name__ == "__main__":
    main()