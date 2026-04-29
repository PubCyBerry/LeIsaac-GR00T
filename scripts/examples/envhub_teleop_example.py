# envhub_teleop_example.py

import logging
import time
import gymnasium as gym

from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.teleoperators import (  
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
    so101_leader,
    bi_so100_leader,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot.envs.factory import make_env


@dataclass
class TeleoperateConfig:
    teleop: TeleoperatorConfig
    env_name: str = "so101_pick_orange"
    fps: int = 60


@dataclass
class EnvWrap:
    env: gym.Env


def make_env_from_leisaac(env_name: str = "so101_pick_orange"):
    envs_dict = make_env(
        f'LightwheelAI/leisaac_env:envs/{env_name}.py',
        n_envs=1,
        trust_remote_code=True
    )
    suite_name = next(iter(envs_dict))
    sync_vector_env = envs_dict[suite_name][0]
    env = sync_vector_env.envs[0].unwrapped

    return env


def teleop_loop(teleop: Teleoperator, env: gym.Env, fps: int):
    from leisaac.devices.action_process import preprocess_device_action
    from leisaac.assets.robots.lerobot import SO101_FOLLOWER_MOTOR_LIMITS
    from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim

    env_wrap = EnvWrap(env=env)

    obs, info = env.reset()
    while True:
        loop_start = time.perf_counter()
        if env.cfg.dynamic_reset_gripper_effort_limit:
            dynamic_reset_gripper_effort_limit_sim(env, 'so101leader')

        raw_action = teleop.get_action()
        processed_action = preprocess_device_action(
            dict(
                so101_leader=True,
                joint_state={
                    k.removesuffix(".pos"): v for k, v in raw_action.items()},
                motor_limits=SO101_FOLLOWER_MOTOR_LIMITS),
            env_wrap
        )
        obs, reward, terminated, truncated, info = env.step(processed_action)
        if terminated or truncated:
            obs, info = env.reset()

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")


def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    teleop = make_teleoperator_from_config(cfg.teleop)
    env = make_env_from_leisaac(cfg.env_name)

    teleop.connect()
    if hasattr(env, 'initialize'):
        env.initialize()
    try:
        teleop_loop(teleop=teleop, env=env, fps=cfg.fps)
    except KeyboardInterrupt:
        pass
    finally:
        teleop.disconnect()
        env.close()


def main():
    teleoperate(TeleoperateConfig(
        teleop=so101_leader.SO101LeaderConfig(
            # port="/dev/ttyACM0",
            port="COM7",
            id='leader',
            use_degrees=False,
        ),
        # env_name="so101_pick_orange",
        # env_name="so101_lift_cube",
        env_name="so101_clean_toytable",
        fps=60,
    ))


if __name__ == "__main__":
    main()
