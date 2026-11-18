"""ippo_mappo.py – обучение группы роботов алгоритмами IPPO и MAPPO."""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from marl_nav_env import MARLNavEnv

LOG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

# Нейросети
def mlp(inp, out, hidden=128):
    return nn.Sequential(
        nn.Linear(inp, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out),
    )

class Actor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = mlp(obs_dim, n_actions, hidden)

    def dist(self, obs):
        return Categorical(logits=self.net(obs))

class Critic(nn.Module):
    def __init__(self, inp_dim, hidden=128):
        super().__init__()
        self.net = mlp(inp_dim, 1, hidden)

    def forward(self, x):
        return self.net(x).squeeze(-1)

class RewardScaler:
    """Нормирует награду на скользящее стандартное отклонение дохода."""

    def __init__(self, shape, gamma):
        self.gamma = gamma
        self.ret = np.zeros(shape)
        self.count, self.mean, self.var = 1e-4, 0.0, 1.0

    def __call__(self, r, done):
        self.ret = self.ret * self.gamma + r
        x = self.ret.ravel()
        b_mean, b_var, b_n = x.mean(), x.var(), x.size
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n
                    + delta ** 2 * self.count * b_n / tot) / tot
        self.count = tot
        self.ret[done] = 0.0
        return r / np.sqrt(self.var + 1e-8)

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)

def critic_input(algo, obs, state, n_agents):
    """Вход критика: наблюдение робота в IPPO, общее состояние в MAPPO."""
    if algo == "ippo":
        return obs
    E = obs.shape[0]
    st = np.repeat(state[:, None, :], n_agents, axis=1)
    ids = np.broadcast_to(np.eye(n_agents, dtype=np.float32), (E, n_agents, n_agents))
    return np.concatenate([st, obs, ids], axis=2).astype(np.float32)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ippo", "mappo"], default="mappo")
    p.add_argument("--reward", choices=["global", "individual", "difference"], default="global")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=400_000, help="всего шагов среды")
    p.add_argument("--n_envs", type=int, default=8, help="параллельных копий среды")
    p.add_argument("--rollout", type=int, default=128, help="шагов на копию за обновление")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--layout", default="warehouse", help="помещение: warehouse (склад) или arena")
    p.add_argument("--n_robots", type=int, default=3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)

    run_name = f"{args.algo}_{args.reward}_s{args.seed}{args.tag}"
    run_dir = os.path.join(LOG_ROOT, args.layout, run_name)
    os.makedirs(run_dir, exist_ok=True)

    E = args.n_envs
    envs = [MARLNavEnv(n_robots=args.n_robots, reward_mode=args.reward, layout=args.layout)
            for _ in range(E)]
    N = envs[0].n
    agents = envs[0].possible_agents
    obs_dim, state_dim = envs[0].obs_dim, envs[0].state_dim
    n_act = envs[0].action_space(agents[0]).n
    c_dim = obs_dim if args.algo == "ippo" else state_dim + obs_dim + N

    actor = Actor(obs_dim, n_act, args.hidden)
    critic = Critic(c_dim, args.hidden)
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, eps=1e-5)
    scaler = RewardScaler((E, N), args.gamma)

    def stack_obs(o):
        return np.stack([o[a] for a in agents])

    obs = np.zeros((E, N, obs_dim), dtype=np.float32)
    state = np.zeros((E, state_dim), dtype=np.float32)
    for k, env in enumerate(envs):
        o, _ = env.reset(seed=args.seed * 1000 + k)
        obs[k] = stack_obs(o)
        state[k] = env.state()

    T = args.rollout
    n_updates = args.steps // (T * E)

    ep_team, ep_own, ep_len, ep_col = (np.zeros(E) for _ in range(4))
    window = []   # последние эпизоды для статистики

    fields = ["update", "env_steps", "episodes", "team_return", "own_return",
              "success_rate", "goals_covered", "collisions", "ep_len",
              "pi_loss", "v_loss", "entropy", "time_s"]
    f = open(os.path.join(run_dir, "progress.csv"), "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    print(f"Запуск {run_name}: {n_updates} обновлений по {T * E} шагов = {n_updates * T * E} шагов среды")
    t0 = time.time()
    total_eps = 0

    for upd in range(1, n_updates + 1):
        for g in opt.param_groups:                       # линейное снижение lr
            g["lr"] = args.lr * (1.0 - (upd - 1) / n_updates)

        b_obs = np.zeros((T, E, N, obs_dim), dtype=np.float32)
        b_cin = np.zeros((T, E, N, c_dim), dtype=np.float32)
        b_act = np.zeros((T, E, N), dtype=np.int64)
        b_logp = np.zeros((T, E, N), dtype=np.float32)
        b_val = np.zeros((T, E, N), dtype=np.float32)
        b_rew = np.zeros((T, E, N), dtype=np.float32)
        b_done = np.zeros((T, E), dtype=np.float32)

        # сбор опыта
        for t in range(T):
            cin = critic_input(args.algo, obs, state, N)
            with torch.no_grad():
                d = actor.dist(torch.as_tensor(obs))
                act = d.sample()
                logp = d.log_prob(act)
                val = critic(torch.as_tensor(cin))
            act_np = act.numpy()

            b_obs[t], b_cin[t], b_act[t] = obs, cin, act_np
            b_logp[t], b_val[t] = logp.numpy(), val.numpy()

            rew = np.zeros((E, N))
            done = np.zeros(E, dtype=bool)
            boot = {}                                    # бутстрэп при обрыве по времени
            for k, env in enumerate(envs):
                o, r, term, trunc, info = env.step(
                    {a: int(act_np[k, i]) for i, a in enumerate(agents)})
                rew[k] = [r[a] for a in agents]
                i0 = info[agents[0]]
                ep_team[k] += i0["global_reward"]
                ep_own[k] += rew[k].mean()
                ep_len[k] += 1
                ep_col[k] += i0["collisions"]
                o_np = stack_obs(o)

                if term[agents[0]] or trunc[agents[0]]:
                    done[k] = True
                    if trunc[agents[0]]:
                        cin_last = critic_input(args.algo, o_np[None], env.state()[None], N)
                        with torch.no_grad():
                            boot[k] = critic(torch.as_tensor(cin_last)).numpy()[0]
                    window.append((ep_team[k], ep_own[k], ep_len[k], float(i0["success"]),
                                   i0["goals_covered"], ep_col[k]))
                    total_eps += 1
                    ep_team[k] = ep_own[k] = ep_len[k] = ep_col[k] = 0
                    o, _ = env.reset()
                    o_np = stack_obs(o)
                obs[k] = o_np
                state[k] = env.state()

            rew_s = scaler(rew, done)
            for k, v_last in boot.items():
                rew_s[k] += args.gamma * v_last
            b_rew[t], b_done[t] = rew_s, done

        # GAE
        with torch.no_grad():
            next_val = critic(torch.as_tensor(critic_input(args.algo, obs, state, N))).numpy()
        adv = np.zeros((T, E, N), dtype=np.float32)
        last = np.zeros((E, N), dtype=np.float32)
        for t in reversed(range(T)):
            nv = next_val if t == T - 1 else b_val[t + 1]
            nonterm = 1.0 - b_done[t][:, None]
            delta = b_rew[t] + args.gamma * nv * nonterm - b_val[t]
            last = delta + args.gamma * args.lam * nonterm * last
            adv[t] = last
        ret = adv + b_val

        # обновление PPO
        B = T * E * N
        f_obs = torch.as_tensor(b_obs.reshape(B, obs_dim))
        f_cin = torch.as_tensor(b_cin.reshape(B, c_dim))
        f_act = torch.as_tensor(b_act.reshape(B))
        f_logp = torch.as_tensor(b_logp.reshape(B))
        f_adv = torch.as_tensor(adv.reshape(B))
        f_ret = torch.as_tensor(ret.reshape(B))
        mb = B // args.minibatches
        pi_l = v_l = ent_l = 0.0
        for _ in range(args.epochs):
            idx = torch.randperm(B)
            for s in range(0, B, mb):
                j = idx[s:s + mb]
                d = actor.dist(f_obs[j])
                new_logp = d.log_prob(f_act[j])
                ratio = torch.exp(new_logp - f_logp[j])
                a = f_adv[j]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pi_loss = -torch.min(ratio * a,
                                     torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a).mean()
                v_loss = ((critic(f_cin[j]) - f_ret[j]) ** 2).mean()
                ent = d.entropy().mean()
                loss = pi_loss + args.vf * v_loss - args.ent * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 0.5)
                opt.step()
                pi_l, v_l, ent_l = pi_loss.item(), v_loss.item(), ent.item()

        # логирование
        recent = window[-50:]
        if recent:
            arr = np.array(recent)
            row = dict(update=upd, env_steps=upd * T * E, episodes=total_eps,
                       team_return=arr[:, 0].mean(), own_return=arr[:, 1].mean(),
                       ep_len=arr[:, 2].mean(), success_rate=arr[:, 3].mean(),
                       goals_covered=arr[:, 4].mean(), collisions=arr[:, 5].mean(),
                       pi_loss=pi_l, v_loss=v_l, entropy=ent_l, time_s=time.time() - t0)
            writer.writerow(row)
            f.flush()
            if upd % 10 == 0 or upd == n_updates:
                print(f"[{upd:4d}/{n_updates}] шагов {row['env_steps']:7d} | "
                      f"команд. награда {row['team_return']:7.1f} | "
                      f"успех {row['success_rate'] * 100:5.1f}% | "
                      f"целей {row['goals_covered']:.2f}/3 | "
                      f"столкн. {row['collisions']:5.1f} | "
                      f"{row['time_s']:5.0f} с", flush=True)
        window = window[-200:]

    f.close()
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "args": vars(args), "obs_dim": obs_dim, "n_actions": n_act},
               os.path.join(run_dir, "model.pt"))
    # Веса актора в формате NumPy – для узла ROS 2 в Gazebo (без PyTorch)
    lin = [m for m in actor.net if isinstance(m, nn.Linear)]
    np.savez(os.path.join(run_dir, "actor.npz"),
             **{f"W{k}": l.weight.detach().numpy() for k, l in enumerate(lin)},
             **{f"b{k}": l.bias.detach().numpy() for k, l in enumerate(lin)})
    print(f"Готово. Модель и лог: {os.path.abspath(run_dir)}")

if __name__ == "__main__":
    main()
