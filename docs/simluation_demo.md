# Test LeIsaac

## 절차

1.환경 설정

```powershell
uv sync
```

2.pick_orange 데이터셋 다운로드

```powershell
hf download --repo-type dataset LightwheelAI/leisaac-pick-orange --local-dir ./datasets/leisaac-so101-pick-orange
```

3. Finetuned GR00T-N1.6-3B 준비

```powershell
git clone --revision e8e625f4f21898c506a1d8f7d20a289c97a52acf https://github.com/NVIDIA/Isaac-GR00T
```

## References

- [LeIsaac Github](https://github.com/lightwheelai/leisaac)
- [LeIsaac - Available Environments](https://lightwheelai.github.io/leisaac/resources/available_env/)
- [LeIsaac × LeRobot EnvHub](https://huggingface.co/docs/lerobot/envhub_leisaac)
- [LeIsaac - so101_pick_orange 데이터셋](https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange)