from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl.environment import PruningEnvironment
from models import get_model
from data import get_dataset
from utils.metrics import evaluate_model_performance

# 加载模型和数据集
model_name = "Qwen2.5-3B"
dataset_name = "wikitext2"
model, tokenizer = get_model(model_name)
dataset = get_dataset(dataset_name)

# 创建评估函数
def evaluate_model(model, dataset, tokenizer):
    return evaluate_model_performance(model, dataset, tokenizer, task_type="language_modeling")

# 创建环境
env = PruningEnvironment(
    model=model,
    tokenizer=tokenizer,
    val_dataset=dataset,
    metric_fn=evaluate_model,
    device="cuda",
    target_sparsity=0.3
)

# 向量化环境(SB3要求)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# 创建并训练模型
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# 保存模型
model.save("ppo_pruning_model")

# 使用模型进行剪枝
obs = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    if dones[0]:
        done = True

print("剪枝完成!")
print(f"最终稀疏度: {info[0]['sparsity']:.4f}")
print(f"最终性能比例: {info[0]['performance_ratio']:.4f}")