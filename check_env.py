import sys, torch, gymnasium as gym, minari
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print(f"Gymnasium: {gym.__version__}")
print(f"Minari: {minari.__version__}")

env = gym.make("Hopper-v5")
obs, _ = env.reset()
print(f"Hopper-v5 OK. obs shape: {obs.shape}, action space: {env.action_space}")
env.close()

remote = minari.list_remote_datasets()
target = "mujoco/hopper/medium-v0"
print(f"{target} available: {target in remote}")
