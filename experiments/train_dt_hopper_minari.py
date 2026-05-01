#!/usr/bin/env python3
"""Train Decision Transformer on Minari Hopper dataset with online evaluation."""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import minari
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decision_transformer import DecisionTransformer, DecisionTransformerConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-id", default="mujoco/hopper/medium-v0")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--num-updates", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--context-len", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-updates", type=int, default=10000)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--final-eval-episodes", type=int, default=100)
    p.add_argument("--target-returns", default="1800,3600")
    p.add_argument("--norm-position", default="pre", choices=["pre", "post"])
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(dataset_id, seed):
    dataset = minari.load_dataset(dataset_id)
    episodes = []
    for ep in dataset.iterate_episodes():
        obs = np.array(ep.observations, dtype=np.float32)
        acts = np.array(ep.actions, dtype=np.float32)
        rews = np.array(ep.rewards, dtype=np.float32)
        terms = np.array(ep.terminations, dtype=bool)
        truncs = np.array(ep.truncations, dtype=bool)
        dones = terms | truncs
        states = obs[:-1]
        rtg = np.zeros_like(rews)
        rtg[-1] = rews[-1]
        for t in reversed(range(len(rews) - 1)):
            rtg[t] = rews[t] + rtg[t + 1]
        episodes.append({
            "states": states,
            "actions": acts,
            "rewards": rews,
            "dones": dones,
            "returns_to_go": rtg,
        })

    n_traj = len(episodes)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_traj)
    n_train = int(0.9 * n_traj)
    train_idx = sorted(indices[:n_train].tolist())
    val_idx = sorted(indices[n_train:].tolist())
    train_eps = [episodes[i] for i in train_idx]
    val_eps = [episodes[i] for i in val_idx]

    all_train_states = np.concatenate([ep["states"] for ep in train_eps], axis=0)
    state_mean = all_train_states.mean(axis=0).astype(np.float32)
    state_std = all_train_states.std(axis=0).astype(np.float32) + 1e-6

    traj_lengths = [len(ep["rewards"]) for ep in episodes]
    traj_returns = [ep["rewards"].sum() for ep in episodes]

    stats = {
        "dataset_id": dataset_id,
        "n_traj": n_traj,
        "n_transitions": sum(traj_lengths),
        "n_train_traj": len(train_eps),
        "n_val_traj": len(val_eps),
        "obs_dim": int(episodes[0]["states"].shape[1]),
        "act_dim": int(episodes[0]["actions"].shape[1]),
        "action_low": [-1.0] * int(episodes[0]["actions"].shape[1]),
        "action_high": [1.0] * int(episodes[0]["actions"].shape[1]),
        "traj_len": {
            "min": int(np.min(traj_lengths)),
            "mean": float(np.mean(traj_lengths)),
            "max": int(np.max(traj_lengths)),
        },
        "traj_return": {
            "min": float(np.min(traj_returns)),
            "mean": float(np.mean(traj_returns)),
            "std": float(np.std(traj_returns)),
            "median": float(np.median(traj_returns)),
            "max": float(np.max(traj_returns)),
            "p90": float(np.percentile(traj_returns, 90)),
            "p95": float(np.percentile(traj_returns, 95)),
        },
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
    }

    return train_eps, val_eps, state_mean, state_std, stats


class TrajectoryDataset:
    def __init__(self, episodes, state_mean, state_std, context_len, max_timestep):
        self.episodes = episodes
        self.state_mean = state_mean
        self.state_std = state_std
        self.context_len = context_len
        self.max_timestep = max_timestep
        self.lengths = np.array([len(ep["rewards"]) for ep in episodes])
        self.weights = self.lengths / self.lengths.sum()

    def sample_batch(self, batch_size, rng):
        K = self.context_len
        ep_indices = rng.choice(len(self.episodes), size=batch_size, p=self.weights)

        states_batch = np.zeros((batch_size, K, self.episodes[0]["states"].shape[1]), dtype=np.float32)
        actions_batch = np.zeros((batch_size, K, self.episodes[0]["actions"].shape[1]), dtype=np.float32)
        rtg_batch = np.zeros((batch_size, K, 1), dtype=np.float32)
        timesteps_batch = np.zeros((batch_size, K), dtype=np.int64)
        mask_batch = np.zeros((batch_size, K), dtype=np.float32)

        for i, ep_idx in enumerate(ep_indices):
            ep = self.episodes[ep_idx]
            ep_len = len(ep["rewards"])
            start = rng.randint(0, ep_len)
            end = min(start + K, ep_len)
            seg_len = end - start

            pad_len = K - seg_len

            s = (ep["states"][start:end] - self.state_mean) / self.state_std
            a = ep["actions"][start:end]
            r = ep["returns_to_go"][start:end] / 1000.0
            ts = np.arange(start, end)
            ts = np.clip(ts, 0, self.max_timestep)

            states_batch[i, pad_len:] = s
            actions_batch[i, pad_len:] = a
            rtg_batch[i, pad_len:, 0] = r
            timesteps_batch[i, pad_len:] = ts
            mask_batch[i, pad_len:] = 1.0

        return {
            "states": torch.from_numpy(states_batch),
            "actions": torch.from_numpy(actions_batch),
            "returns_to_go": torch.from_numpy(rtg_batch),
            "timesteps": torch.from_numpy(timesteps_batch),
            "attention_mask": torch.from_numpy(mask_batch),
        }


def run_sanity_checks(model, train_dataset, device, rng):
    results = {}
    K = model.config.context_len
    B = 4
    sd = model.config.state_dim
    ad = model.config.act_dim

    batch = train_dataset.sample_batch(B, rng)
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(batch["states"], batch["actions"], batch["returns_to_go"],
                batch["timesteps"], batch["attention_mask"])

    ap_shape = list(out["action_preds"].shape)
    sp_shape = list(out["state_preds"].shape)
    rp_shape = list(out["return_preds"].shape)
    shapes_ok = (ap_shape == [B, K, ad] and sp_shape == [B, K, sd] and rp_shape == [B, K, 1])
    results["forward_shapes"] = {
        "action_preds": ap_shape,
        "state_preds": sp_shape,
        "return_preds": rp_shape,
        "expected_action": [B, K, ad],
        "expected_state": [B, K, sd],
        "expected_return": [B, K, 1],
        "pass": shapes_ok,
    }

    loss = DecisionTransformer.masked_action_mse(
        out["action_preds"], batch["actions"], batch["attention_mask"]
    )
    loss.backward()
    grads_finite = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.grad is not None
    )
    results["loss_grad_finite"] = {
        "loss": float(loss.item()),
        "loss_finite": bool(torch.isfinite(loss)),
        "grads_finite": grads_finite,
        "pass": bool(torch.isfinite(loss)) and grads_finite,
    }
    model.zero_grad()

    n_overfit = min(4, len(train_dataset.episodes))
    overfit_eps = train_dataset.episodes[:n_overfit]
    overfit_ds = TrajectoryDataset(overfit_eps, train_dataset.state_mean,
                                   train_dataset.state_std, train_dataset.context_len,
                                   train_dataset.max_timestep)
    overfit_model = DecisionTransformer(model.config).to(device)
    overfit_opt = torch.optim.AdamW(overfit_model.parameters(), lr=3e-4, weight_decay=0)
    overfit_rng = np.random.RandomState(42)

    overfit_model.train()
    initial_loss = None
    final_loss = None
    for step in range(500):
        ob = overfit_ds.sample_batch(min(8, n_overfit), overfit_rng)
        ob = {k: v.to(device) for k, v in ob.items()}
        out_o = overfit_model(ob["states"], ob["actions"], ob["returns_to_go"],
                             ob["timesteps"], ob["attention_mask"])
        l = DecisionTransformer.masked_action_mse(
            out_o["action_preds"], ob["actions"], ob["attention_mask"]
        )
        if step == 0:
            initial_loss = l.item()
        overfit_opt.zero_grad()
        l.backward()
        overfit_opt.step()
        final_loss = l.item()

    pct_decrease = (initial_loss - final_loss) / (initial_loss + 1e-8) * 100
    results["tiny_overfit"] = {
        "n_trajectories": n_overfit,
        "n_updates": 500,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "pct_decrease": pct_decrease,
        "pass": pct_decrease > 50,
    }

    results["all_pass"] = all(r["pass"] for r in results.values())
    return results


def evaluate_policy(model, state_mean, state_std, target_returns, n_episodes, device, seed):
    state_mean_t = torch.from_numpy(state_mean).to(device)
    state_std_t = torch.from_numpy(state_std).to(device)
    K = model.config.context_len
    act_dim = model.config.act_dim

    all_results = []
    for target in target_returns:
        returns = []
        lengths = []
        for ep_i in range(n_episodes):
            env = gym.make("Hopper-v5")
            obs, _ = env.reset(seed=seed + ep_i)
            done = False
            total_reward = 0.0
            ep_len = 0

            states_list = []
            actions_list = []
            rtg_list = []
            timesteps_list = []

            rtg_value = target / 1000.0

            while not done:
                state = torch.from_numpy(obs).float()
                states_list.append(state)
                actions_list.append(torch.zeros(act_dim))
                rtg_list.append(torch.tensor([rtg_value], dtype=torch.float32))
                timesteps_list.append(torch.tensor(min(ep_len, model.config.max_timestep), dtype=torch.long))

                states_t = torch.stack(states_list)
                actions_t = torch.stack(actions_list)
                rtg_t = torch.stack(rtg_list)
                timesteps_t = torch.stack(timesteps_list)

                action = model.get_action(
                    states_t, actions_t, rtg_t, timesteps_t,
                    state_mean=state_mean_t, state_std=state_std_t,
                )
                action_np = action.cpu().numpy()
                action_np = np.clip(action_np, -1.0, 1.0)

                actions_list[-1] = torch.from_numpy(action_np).float()

                obs, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated
                total_reward += reward
                rtg_value -= reward / 1000.0
                ep_len += 1

            env.close()
            returns.append(total_reward)
            lengths.append(ep_len)

        returns = np.array(returns)
        lengths = np.array(lengths)
        all_results.append({
            "target_return": target,
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std()),
            "min_return": float(returns.min()),
            "max_return": float(returns.max()),
            "median_return": float(np.median(returns)),
            "mean_length": float(lengths.mean()),
            "std_length": float(lengths.std()),
            "n_episodes": n_episodes,
        })
    return all_results


def evaluate_random_policy(n_episodes, seed):
    returns = []
    lengths = []
    for ep_i in range(n_episodes):
        env = gym.make("Hopper-v5")
        obs, _ = env.reset(seed=seed + 10000 + ep_i)
        done = False
        total_reward = 0.0
        ep_len = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            ep_len += 1
        env.close()
        returns.append(total_reward)
        lengths.append(ep_len)
    returns = np.array(returns)
    lengths = np.array(lengths)
    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "median_return": float(np.median(returns)),
        "mean_length": float(lengths.mean()),
        "std_length": float(lengths.std()),
        "n_episodes": n_episodes,
    }


class TeeLogger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    args = parse_args()

    if args.smoke:
        args.num_updates = 500
        args.eval_episodes = 3
        args.final_eval_episodes = 3
        args.warmup_updates = 100
        args.eval_every = 250

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    if args.run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = f"runs/dt_hopper_minari/{timestamp}"
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    sys.stdout = TeeLogger(run_dir / "run.log")

    target_returns = [float(x) for x in args.target_returns.split(",")]
    set_seed(args.seed)

    print("Loading dataset...")
    train_eps, val_eps, state_mean, state_std, dataset_stats = load_dataset(
        args.dataset_id, args.seed
    )
    obs_dim = dataset_stats["obs_dim"]
    act_dim = dataset_stats["act_dim"]

    with open(run_dir / "dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=2)

    config = DecisionTransformerConfig(
        state_dim=obs_dim,
        act_dim=act_dim,
        context_len=args.context_len,
        norm_type=args.norm_position,
    )
    model = DecisionTransformer(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    config_dict = {
        "state_dim": config.state_dim,
        "act_dim": config.act_dim,
        "hidden_size": config.hidden_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "context_len": config.context_len,
        "max_timestep": config.max_timestep,
        "dropout": config.dropout,
        "norm_position": args.norm_position,
        "action_tanh": config.action_tanh,
        "num_updates": args.num_updates,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_updates": args.warmup_updates,
        "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes,
        "final_eval_episodes": args.final_eval_episodes,
        "target_returns": target_returns,
        "seed": args.seed,
        "device": args.device,
        "smoke": args.smoke,
        "param_count": param_count,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    train_rng = np.random.RandomState(args.seed)
    val_rng = np.random.RandomState(args.seed + 1)

    train_dataset = TrajectoryDataset(train_eps, state_mean, state_std,
                                      args.context_len, config.max_timestep)
    val_dataset = TrajectoryDataset(val_eps, state_mean, state_std,
                                    args.context_len, config.max_timestep)

    print("Running sanity checks...")
    sanity_rng = np.random.RandomState(args.seed + 99)
    sanity_results = run_sanity_checks(model, train_dataset, device, sanity_rng)
    with open(run_dir / "sanity_checks.json", "w") as f:
        json.dump(sanity_results, f, indent=2)
    print(f"Sanity checks: {'PASS' if sanity_results['all_pass'] else 'FAIL'}")
    if not sanity_results["all_pass"]:
        print("Sanity checks failed. Aborting.")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step):
        if step < args.warmup_updates:
            return step / max(1, args.warmup_updates)
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)

    print("Evaluating random policy...")
    random_eval = evaluate_random_policy(max(20, args.eval_episodes), args.seed)
    with open(run_dir / "random_policy_eval.json", "w") as f:
        json.dump(random_eval, f, indent=2)
    print(f"Random policy mean return: {random_eval['mean_return']:.1f}")

    metrics_log = []
    eval_log = []
    best_score = -float("inf")
    start_time = time.time()

    eval_updates = set([0] + list(range(args.eval_every, args.num_updates, args.eval_every)) + [args.num_updates])

    diverged = False
    initial_train_mse = None
    last_eval_3600 = None
    last_eval_1800 = None
    last_train_mse = None
    last_val_mse = None

    print("Starting training...")
    for update in tqdm(range(args.num_updates + 1), desc="Training"):
        if update in eval_updates:
            n_ep = args.final_eval_episodes if update == args.num_updates else args.eval_episodes
            model.eval()
            eval_results = evaluate_policy(
                model, state_mean, state_std, target_returns, n_ep, device, args.seed
            )
            for r in eval_results:
                r["update"] = update
                eval_log.append(r)
                if r["target_return"] == 3600:
                    last_eval_3600 = r["mean_return"]
                    if r["mean_return"] > best_score:
                        best_score = r["mean_return"]
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config": config_dict,
                            "update": update,
                            "state_mean": state_mean.tolist(),
                            "state_std": state_std.tolist(),
                            "best_score": best_score,
                            "seed": args.seed,
                        }, run_dir / "checkpoint_best.pt")
                if r["target_return"] == 1800:
                    last_eval_1800 = r["mean_return"]
            model.train()

            elapsed = time.time() - start_time
            elapsed_str = f"{elapsed/60:.1f}m" if elapsed < 3600 else f"{elapsed/3600:.1f}h"
            t_mse_str = f"{last_train_mse:.4f}" if last_train_mse is not None else "N/A"
            v_mse_str = f"{last_val_mse:.4f}" if last_val_mse is not None else "N/A"
            e3600_str = f"{last_eval_3600:.1f}" if last_eval_3600 is not None else "N/A"
            e1800_str = f"{last_eval_1800:.1f}" if last_eval_1800 is not None else "N/A"
            print(f"\n[STATUS] update {update}/{args.num_updates}, train_mse={t_mse_str}, val_mse={v_mse_str}, eval_3600={e3600_str}, eval_1800={e1800_str}, elapsed={elapsed_str}")

            pd.DataFrame(eval_log).to_csv(run_dir / "eval_returns.csv", index=False)
            if metrics_log:
                pd.DataFrame(metrics_log).to_csv(run_dir / "metrics.csv", index=False)

        if update == args.num_updates:
            break

        batch = train_dataset.sample_batch(args.batch_size, train_rng)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["states"], batch["actions"], batch["returns_to_go"],
                    batch["timesteps"], batch["attention_mask"])
        loss = DecisionTransformer.masked_action_mse(
            out["action_preds"], batch["actions"], batch["attention_mask"]
        )

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        scheduler.step()

        has_nan = not torch.isfinite(loss)
        current_mse = float(loss.item())
        last_train_mse = current_mse

        if initial_train_mse is None:
            initial_train_mse = current_mse

        if has_nan or current_mse > initial_train_mse * 10:
            print(f"\n[DIVERGENCE DETECTED] update {update}, train_mse={current_mse}, initial_mse={initial_train_mse}, NaN={has_nan}")
            print("Stopping early to avoid wasting compute.")
            diverged = True
            break

        if update % 100 == 0:
            val_batch = val_dataset.sample_batch(args.batch_size, val_rng)
            val_batch = {k: v.to(device) for k, v in val_batch.items()}
            with torch.no_grad():
                val_out = model(val_batch["states"], val_batch["actions"],
                               val_batch["returns_to_go"], val_batch["timesteps"],
                               val_batch["attention_mask"])
                val_loss = DecisionTransformer.masked_action_mse(
                    val_out["action_preds"], val_batch["actions"], val_batch["attention_mask"]
                )

            last_val_mse = float(val_loss.item())
            metrics_log.append({
                "update": update,
                "train_mse": current_mse,
                "val_mse": last_val_mse,
                "lr": float(scheduler.get_last_lr()[0]),
                "grad_norm": float(grad_norm),
                "param_count": param_count,
                "elapsed_s": time.time() - start_time,
                "nan_inf": bool(has_nan),
            })

    final_update = update if diverged else args.num_updates

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config_dict,
        "update": final_update,
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "best_score": best_score,
        "seed": args.seed,
        "diverged": diverged,
    }, run_dir / "checkpoint_latest.pt")

    metrics_df = pd.DataFrame(metrics_log)
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)

    eval_df = pd.DataFrame(eval_log)
    eval_df.to_csv(run_dir / "eval_returns.csv", index=False)

    if len(metrics_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(metrics_df["update"], metrics_df["train_mse"], label="train")
        axes[0].plot(metrics_df["update"], metrics_df["val_mse"], label="val")
        axes[0].set_xlabel("Update")
        axes[0].set_ylabel("Action MSE")
        axes[0].set_title("Loss Curve")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].axis("off")
        plt.tight_layout()
        fig.savefig(run_dir / "loss_curve.png", dpi=100)
        plt.close(fig)

    if len(eval_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        for target in target_returns:
            sub = eval_df[eval_df["target_return"] == target]
            if len(sub) > 0:
                ax.plot(sub["update"], sub["mean_return"], marker="o", label=f"target={int(target)}")
                ax.fill_between(sub["update"],
                                sub["mean_return"] - sub["std_return"],
                                sub["mean_return"] + sub["std_return"], alpha=0.2)
        ax.axhline(random_eval["mean_return"], color="gray", linestyle="--", label="random")
        ax.set_xlabel("Update")
        ax.set_ylabel("Mean Return")
        ax.set_title("Eval Return Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(run_dir / "eval_return_curve.png", dpi=100)
        plt.close(fig)

    last_update_logged = final_update
    final_eval_1800 = [r for r in eval_log if r["update"] == last_update_logged and r["target_return"] == 1800]
    final_eval_3600 = [r for r in eval_log if r["update"] == last_update_logged and r["target_return"] == 3600]
    if not final_eval_1800:
        final_eval_1800 = [r for r in eval_log if r["target_return"] == 1800][-1:] if any(r["target_return"] == 1800 for r in eval_log) else []
    if not final_eval_3600:
        final_eval_3600 = [r for r in eval_log if r["target_return"] == 3600][-1:] if any(r["target_return"] == 3600 for r in eval_log) else []
    best_eval_1800 = max([r for r in eval_log if r["target_return"] == 1800], key=lambda x: x["mean_return"]) if any(r["target_return"] == 1800 for r in eval_log) else {"mean_return": float("nan")}
    best_eval_3600 = max([r for r in eval_log if r["target_return"] == 3600], key=lambda x: x["mean_return"]) if any(r["target_return"] == 3600 for r in eval_log) else {"mean_return": float("nan")}

    final_train_mse = metrics_log[-1]["train_mse"] if metrics_log else float("nan")
    final_val_mse = metrics_log[-1]["val_mse"] if metrics_log else float("nan")

    loss_decreased = (metrics_log[-1]["train_mse"] < metrics_log[0]["train_mse"]) if len(metrics_log) > 1 else False
    above_random = best_eval_3600["mean_return"] > random_eval["mean_return"] * 1.5

    if diverged:
        conclusion = "FAIL"
    elif sanity_results["all_pass"] and loss_decreased and above_random:
        conclusion = "PASS"
    elif sanity_results["all_pass"] and loss_decreased:
        conclusion = "PARTIAL"
    else:
        conclusion = "FAIL"

    final_1800_str = f"{final_eval_1800[0]['mean_return']:.1f}" if final_eval_1800 else "N/A"
    final_3600_str = f"{final_eval_3600[0]['mean_return']:.1f}" if final_eval_3600 else "N/A"
    ds_mean_ret = dataset_stats["traj_return"]["mean"]
    ds_p95_ret = dataset_stats["traj_return"]["p95"]
    rand_mean = random_eval["mean_return"]

    if conclusion == "PASS":
        conclusion_text = "Training converged and online policy clearly outperforms random."
    elif conclusion == "PARTIAL":
        conclusion_text = "Training converged but online policy performance is weak/unstable."
    else:
        conclusion_text = "Training failed — check sanity checks and loss curves."

    summary = f"""# Decision Transformer — Hopper Minari Training Summary

| metric | value |
|--------|-------|
| parameter count | {param_count:,} |
| norm_position | {args.norm_position} |
| train trajectories | {dataset_stats['n_train_traj']} |
| validation trajectories | {dataset_stats['n_val_traj']} |
| total transitions | {dataset_stats['n_transitions']:,} |
| dataset mean return | {ds_mean_ret:.1f} |
| dataset 95th percentile return | {ds_p95_ret:.1f} |
| random policy mean return | {rand_mean:.1f} |
| best eval mean return, target 1800 | {best_eval_1800['mean_return']:.1f} |
| best eval mean return, target 3600 | {best_eval_3600['mean_return']:.1f} |
| final eval mean return, target 1800 | {final_1800_str} |
| final eval mean return, target 3600 | {final_3600_str} |
| final train action MSE | {final_train_mse:.6f} |
| final validation action MSE | {final_val_mse:.6f} |

## Conclusion: **{conclusion}**

{conclusion_text}
"""
    with open(run_dir / "summary.md", "w") as f:
        f.write(summary)

    print(f"\nTraining complete. Conclusion: {conclusion}")
    print(f"Artifacts saved to: {run_dir}")
    print(f"Best eval return (target 3600): {best_score:.1f}")
    print(f"Random policy return: {random_eval['mean_return']:.1f}")


if __name__ == "__main__":
    main()
