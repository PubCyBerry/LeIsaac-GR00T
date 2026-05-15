#!/usr/bin/env python
"""policy-client-shim.py — Workaround for huggingface/lerobot#3078.

lerobot 0.4.4 의 `lerobot/async_inference/robot_client.py` 는 import 블록에서
built-in robot config 모듈들 (`so_follower`, `bi_so_follower`, `koch_follower`,
`omx_follower`) 을 import 하지 않는다. 결과적으로 `RobotConfig.register_subclass`
데코레이터가 실행되지 않아 draccus choice registry 가 비어 있고,
``--robot.type=so101_follower`` 같은 인자가 ``invalid choice ... (choose from )`` 로 거부된다.

해당 회귀는 upstream PR #3081 에서 수정되었으나 0.4.4 태그에는 미반영. 본 shim 은
robot_client 의 ``__main__`` 을 실행하기 전에 필요한 robot config 모듈을 선행
import 해 registry 를 채운 뒤 runpy 로 원래 진입점을 그대로 호출한다.

upstream 패치가 들어간 lerobot 버전으로 올라가면 본 파일을 삭제하고
``entrypoint.sh`` 의 ``policy-client`` 분기가 ``python -m
lerobot.async_inference.robot_client`` 를 직접 호출하도록 되돌리면 된다.
"""

# ── Built-in robot configs — side-effect imports (PR #3081 와 동일 목록) ────
# 각 모듈의 모듈-레벨 코드가 ``@RobotConfig.register_subclass(...)`` 를 실행해
# choice registry 에 항목을 추가한다.
import lerobot.robots.so_follower.config_so_follower  # noqa: F401  (so101_follower, so100_follower)
import lerobot.robots.bi_so_follower.config_bi_so_follower  # noqa: F401
import lerobot.robots.koch_follower.config_koch_follower  # noqa: F401
import lerobot.robots.omx_follower.config_omx_follower  # noqa: F401

import runpy

# robot_client 의 ``if __name__ == "__main__":`` 블록을 그대로 실행한다.
# alter_sys=True 로 sys.argv[0] / sys.modules['__main__'] 가 robot_client 를
# 가리키게 해 argparse 가 정상 동작하도록 한다.
runpy.run_module(
    "lerobot.async_inference.robot_client",
    run_name="__main__",
    alter_sys=True,
)
