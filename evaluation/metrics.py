"""
Evaluation module:
- Exploitability computation (Nash convergence metric)
- Head-to-head simulation between strategies
- Visualization helpers (returns matplotlib figures as base64 PNGs)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import io, base64
from typing import List, Dict, Callable, Optional, Tuple
from game_engine.games import GameState


# ─────────────────────────────────────────────
#  Best Response & Exploitability
# ─────────────────────────────────────────────

def best_response_value(
    state: GameState,
    br_player: int,
    get_strategy: Callable[[GameState, int], Dict[int, float]],
    reach: Optional[List[float]] = None,
) -> float:
    """
    Compute best-response value for br_player against the given strategy profile.
    Returns: expected payoff for br_player.
    """
    if reach is None:
        reach = [1.0] * state.num_players()

    if state.is_terminal():
        return state.returns()[br_player]

    player  = state.current_player()
    actions = state.legal_actions()

    action_values = {}
    for a in actions:
        new_reach = reach.copy()
        child     = state.apply_action(a)
        if player != br_player:
            strat = get_strategy(state, player)
            p = strat.get(a, 1.0 / len(actions))
            new_reach[player] *= p
        action_values[a] = best_response_value(child, br_player, get_strategy, new_reach)

    if player == br_player:
        # BR player picks the best action
        return max(action_values.values())
    else:
        # Other players follow their current strategy
        strat = get_strategy(state, player)
        return sum(strat.get(a, 1.0 / len(actions)) * action_values[a] for a in actions)


def compute_exploitability(
    game_factory: Callable,
    get_strategy: Callable[[GameState, int], Dict[int, float]],
    num_samples: int = 20,
) -> float:
    """
    Monte Carlo estimate of exploitability (averaged over game instances).
    Lower = closer to Nash. At NE = 0.
    """
    total = 0.0
    for s in range(num_samples):
        state = game_factory(seed=s * 7)
        n = state.num_players()
        br_gains = []
        for p in range(n):
            br_val  = best_response_value(state, p, get_strategy)
            # Approximate Nash value by averaging strategy value
            avg_val = _strategy_value(state, p, get_strategy)
            br_gains.append(max(0.0, br_val - avg_val))
        total += sum(br_gains) / n
    return total / num_samples


def _strategy_value(
    state: GameState,
    player: int,
    get_strategy: Callable,
) -> float:
    if state.is_terminal():
        return state.returns()[player]
    curr = state.current_player()
    strat = get_strategy(state, curr)
    actions = state.legal_actions()
    return sum(
        strat.get(a, 1.0 / len(actions)) * _strategy_value(state.apply_action(a), player, get_strategy)
        for a in actions
    )


# ─────────────────────────────────────────────
#  Head-to-head simulation
# ─────────────────────────────────────────────

def simulate_game(
    game_factory: Callable,
    strategies: List[Callable[[GameState, int], Dict[int, float]]],
    num_games: int = 500,
    seed: int = 0,
) -> Dict:
    """
    Run num_games with the given strategy per player.
    Returns per-player average payoff and win rates.
    """
    rng = np.random.default_rng(seed)
    n_players = None
    all_returns = []

    for g in range(num_games):
        state = game_factory(seed=int(rng.integers(0, 100000)))
        if n_players is None:
            n_players = state.num_players()

        while not state.is_terminal():
            p       = state.current_player()
            actions = state.legal_actions()
            strat   = strategies[p](state, p)
            probs   = np.array([strat.get(a, 1.0 / len(actions)) for a in actions])
            probs   = np.clip(probs, 0, None)
            if probs.sum() < 1e-10:
                probs = np.ones_like(probs)
            probs /= probs.sum()
            chosen  = actions[np.random.choice(len(actions), p=probs)]
            state   = state.apply_action(chosen)

        all_returns.append(state.returns())

    returns_arr = np.array(all_returns)
    return {
        "mean_returns":    returns_arr.mean(axis=0).tolist(),
        "std_returns":     returns_arr.std(axis=0).tolist(),
        "win_rates":       (returns_arr > 0).mean(axis=0).tolist(),
        "num_games":       num_games,
    }


# ─────────────────────────────────────────────
#  Plotting helpers → return base64 PNG strings
# ─────────────────────────────────────────────

COLORS = ["#6366f1", "#f97316", "#10b981", "#f59e0b", "#ef4444"]

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140, facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def plot_exploitability_curves(
    curves: List[Tuple[str, List[Tuple[int, float]]]],
    title: str = "Exploitability vs. Iterations",
) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("#0f0f11")
    ax.set_facecolor("#161618")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    for i, (label, data) in enumerate(curves):
        xs = [d[0] for d in data]
        ys = [d[1] for d in data]
        ax.plot(xs, ys, color=COLORS[i % len(COLORS)], linewidth=2, label=label, marker="o", markersize=3)

    ax.set_xlabel("Iterations", color="#aaaaaa", fontsize=11)
    ax.set_ylabel("Exploitability", color="#aaaaaa", fontsize=11)
    ax.set_title(title, color="#eeeeee", fontsize=13, fontweight="bold", pad=14)
    ax.legend(facecolor="#222222", edgecolor="#444444", labelcolor="#dddddd")
    ax.grid(True, color="#2a2a2a", linewidth=0.5)
    plt.tight_layout()
    result = _fig_to_b64(fig)
    plt.close(fig)
    return result


def plot_strategy_heatmap(
    strategies: Dict[str, Dict[int, float]],
    action_labels: List[str],
    title: str = "Average Strategy Heatmap",
) -> str:
    """Show strategy probabilities across info sets."""
    keys  = list(strategies.keys())[:20]  # limit to 20 rows
    n_act = len(action_labels)
    matrix = np.array([[strategies[k].get(i, 0.0) for i in range(n_act)] for k in keys])

    fig, ax = plt.subplots(figsize=(max(5, n_act * 1.2), max(4, len(keys) * 0.35 + 1.5)))
    fig.patch.set_facecolor("#0f0f11")
    ax.set_facecolor("#161618")

    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_act))
    ax.set_xticklabels(action_labels, color="#aaaaaa")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([k[:30] for k in keys], color="#aaaaaa", fontsize=7)
    ax.set_title(title, color="#eeeeee", fontsize=12, fontweight="bold", pad=12)
    plt.colorbar(im, ax=ax, label="Probability").ax.yaxis.label.set_color("#aaaaaa")
    plt.tight_layout()
    result = _fig_to_b64(fig)
    plt.close(fig)
    return result


def plot_training_loss(log: List[Dict], title: str = "Training Loss") -> str:
    iters = [e["iteration"] for e in log]
    strat_losses = [e["strat_loss"] for e in log]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("#0f0f11")

    # Advantage losses per player
    ax = axes[0]
    ax.set_facecolor("#161618")
    for spine in ax.spines.values(): spine.set_edgecolor("#333333")
    n_players = len(log[0]["adv_losses"])
    for p in range(n_players):
        vals = [e["adv_losses"][p] for e in log]
        ax.plot(iters, vals, color=COLORS[p], linewidth=1.5, label=f"Player {p}")
    ax.set_xlabel("Iteration", color="#aaaaaa"); ax.set_ylabel("Loss", color="#aaaaaa")
    ax.set_title("Advantage Network Loss", color="#eeeeee", fontsize=11)
    ax.legend(facecolor="#222222", edgecolor="#444444", labelcolor="#dddddd")
    ax.tick_params(colors="#aaaaaa")
    ax.grid(True, color="#2a2a2a", linewidth=0.5)

    # Strategy loss
    ax = axes[1]
    ax.set_facecolor("#161618")
    for spine in ax.spines.values(): spine.set_edgecolor("#333333")
    ax.plot(iters, strat_losses, color=COLORS[2], linewidth=1.5)
    ax.set_xlabel("Iteration", color="#aaaaaa"); ax.set_ylabel("Loss", color="#aaaaaa")
    ax.set_title("Strategy Network Loss", color="#eeeeee", fontsize=11)
    ax.tick_params(colors="#aaaaaa")
    ax.grid(True, color="#2a2a2a", linewidth=0.5)

    fig.suptitle(title, color="#eeeeee", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    result = _fig_to_b64(fig)
    plt.close(fig)
    return result


def plot_payoff_bars(sim_results: Dict, player_labels: List[str], title: str = "Payoffs") -> str:
    means = sim_results["mean_returns"]
    stds  = sim_results["std_returns"]
    n     = len(means)

    fig, ax = plt.subplots(figsize=(max(5, n * 1.5), 4))
    fig.patch.set_facecolor("#0f0f11")
    ax.set_facecolor("#161618")
    for spine in ax.spines.values(): spine.set_edgecolor("#333333")

    bars = ax.bar(range(n), means, color=COLORS[:n], alpha=0.85, width=0.5,
                  yerr=stds, capsize=5, error_kw={"ecolor": "#888888", "linewidth": 1})
    ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(n))
    ax.set_xticklabels(player_labels, color="#aaaaaa", fontsize=11)
    ax.set_ylabel("Expected payoff", color="#aaaaaa")
    ax.set_title(title, color="#eeeeee", fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors="#aaaaaa")
    ax.grid(True, axis="y", color="#2a2a2a", linewidth=0.5)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{mean:.2f}", ha="center", va="bottom", color="#dddddd", fontsize=9)
    plt.tight_layout()
    result = _fig_to_b64(fig)
    plt.close(fig)
    return result
