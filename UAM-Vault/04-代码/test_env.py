import torch
import vmas
from vmas import make_env

# 1. 验证 PyTorch GPU
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 2. 验证 VMAS
print(f"VMAS: {vmas.__version__}")

# 3. 创建简单环境测试
print("Creating VMAS environment...")
env = make_env(
    scenario="navigation",
    num_envs=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
    continuous_actions=True,
)
obs = env.reset()
print(f"Num agents: {env.n_agents}")
print(f"Observation shapes: {[o.shape for o in obs]}")
print("All tests passed!")
