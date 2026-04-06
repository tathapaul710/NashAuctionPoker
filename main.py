"""
Main experiment runner.
Trains both Vanilla CFR and Deep CFR on Leduc Hold'em and
the Asymmetric Auction game, then produces a JSON results file.
"""

import json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from game_engine.games import LeducState, AuctionState
from cfr.vanilla_cfr import VanillaCFR
from cfr.deep_cfr import DeepCFR
from evaluation.metrics import (
    compute_exploitability, simulate_game,
    plot_exploitability_curves, plot_strategy_heatmap,
    plot_training_loss, plot_payoff_bars
)


# ─────────────────────────────────────────────────────
#  Experiment 1: Vanilla CFR on Leduc Hold'em
# ─────────────────────────────────────────────────────

def run_vanilla_cfr_leduc(n_iters: int = 500):
    print("\n=== Vanilla CFR on Leduc Hold'em ===")
    solver = VanillaCFR(game_factory=LeducState.new_game)
    t0 = time.time()
    expl_curve = solver.train(n_iters)
    elapsed = time.time() - t0

    # Collect sample strategies
    state = LeducState.new_game(seed=0)
    sample_strategies = {}
    for p in range(2):
        key = state.info_set_key(p)
        strat = solver.get_policy(state, p)
        if strat:
            sample_strategies[key] = strat

    print(f"  Done in {elapsed:.1f}s | Final exploitability: {expl_curve[-1][1]:.4f}")
    return {
        "exploitability_curve": expl_curve,
        "sample_strategies": sample_strategies,
        "elapsed": elapsed,
    }


# ─────────────────────────────────────────────────────
#  Experiment 2: Deep CFR on Leduc Hold'em
# ─────────────────────────────────────────────────────

def run_deep_cfr_leduc(n_iters: int = 20, traversals: int = 80):
    print("\n=== Deep CFR on Leduc Hold'em ===")
    solver = DeepCFR(
        game_factory     = LeducState.new_game,
        state_dim        = 30,
        num_actions      = 3,
        num_players      = 2,
        buffer_capacity  = 50_000,
        hidden_dim       = 64,
        lr               = 1e-3,
        train_batch      = 256,
        train_steps      = 100,
    )
    t0 = time.time()
    log = solver.train(n_iters, traversals_per_iter=traversals)
    elapsed = time.time() - t0

    # Simulate vs random strategy
    def random_strategy(state, player):
        acts = state.legal_actions()
        return {a: 1.0/len(acts) for a in acts}

    def deep_cfr_strategy(state, player):
        return solver.get_policy(state, player)

    sim = simulate_game(
        game_factory=LeducState.new_game,
        strategies=[deep_cfr_strategy, random_strategy],
        num_games=200,
    )
    solver.save_model("leduc_model.pth")
    print(f"  Saved Leduc strategy model to leduc_model.pth")
    print(f"  Done in {elapsed:.1f}s | vs random: {sim['mean_returns']}")
    return {
        "training_log": log,
        "sim_vs_random": sim,
        "elapsed": elapsed,
    }


# ─────────────────────────────────────────────────────
#  Experiment 3: Deep CFR on Asymmetric Auction (3-player)
# ─────────────────────────────────────────────────────

def run_deep_cfr_auction(n_iters: int = 15, traversals: int = 60):
    print("\n=== Deep CFR on Asymmetric 3-Player Auction ===")
    solver = DeepCFR(
        game_factory     = AuctionState.new_game,
        state_dim        = 26,   # updated: vec size 26 after IndexError fix
        num_actions      = 5,   # max action space (seller has 5 price options)
        num_players      = 3,
        buffer_capacity  = 30_000,
        hidden_dim       = 64,
        lr               = 1e-3,
        train_batch      = 128,
        train_steps      = 80,
    )
    t0 = time.time()
    log = solver.train(n_iters, traversals_per_iter=traversals)
    elapsed = time.time() - t0

    def deep_cfr_strategy(state, player):
        return solver.get_policy(state, player)

    def random_strategy(state, player):
        acts = state.legal_actions()
        return {a: 1.0/len(acts) for a in acts}

    # CFR vs all-random
    sim_cfr_vs_random = simulate_game(
        game_factory=AuctionState.new_game,
        strategies=[deep_cfr_strategy, random_strategy, random_strategy],
        num_games=200,
    )
    # All CFR
    sim_all_cfr = simulate_game(
        game_factory=AuctionState.new_game,
        strategies=[deep_cfr_strategy, deep_cfr_strategy, deep_cfr_strategy],
        num_games=200,
    )

    print(f"  Done in {elapsed:.1f}s")
    print(f"  CFR seller vs random buyers: {sim_cfr_vs_random['mean_returns']}")
    print(f"  All CFR equilibrium payoffs: {sim_all_cfr['mean_returns']}")
    return {
        "training_log": log,
        "sim_cfr_vs_random": sim_cfr_vs_random,
        "sim_all_cfr": sim_all_cfr,
        "elapsed": elapsed,
    }


# ─────────────────────────────────────────────────────
#  Generate all plots and bundle results
# ─────────────────────────────────────────────────────

def run_all(quick: bool = False):
    """
    quick=True → fewer iterations for fast demo.
    quick=False → full training run.
    """
    iters_vcfr    = 100  if quick else 500
    iters_dcfr    = 10   if quick else 20
    iters_auction = 8    if quick else 15

    results = {}

    # --- Vanilla CFR ---
    r1 = run_vanilla_cfr_leduc(iters_vcfr)
    results["vanilla_cfr"] = {
        "exploitability_curve": r1["exploitability_curve"],
        "elapsed": r1["elapsed"],
    }

    # --- Deep CFR Leduc ---
    r2 = run_deep_cfr_leduc(iters_dcfr, traversals=50 if quick else 80)
    results["deep_cfr_leduc"] = {
        "training_log": r2["training_log"],
        "sim_vs_random": r2["sim_vs_random"],
        "elapsed": r2["elapsed"],
    }

    # --- Deep CFR Auction ---
    r3 = run_deep_cfr_auction(iters_auction, traversals=40 if quick else 60)
    results["deep_cfr_auction"] = {
        "training_log": r3["training_log"],
        "sim_cfr_vs_random": r3["sim_cfr_vs_random"],
        "sim_all_cfr": r3["sim_all_cfr"],
        "elapsed": r3["elapsed"],
    }

    # --- Plots ---
    print("\n=== Generating plots ===")

    # Exploitability curve
    results["plots"] = {}
    vcfr_curve = r1["exploitability_curve"]
    results["plots"]["exploitability"] = plot_exploitability_curves(
        [("Vanilla CFR", vcfr_curve)],
        title="Vanilla CFR: Exploitability on Leduc Hold'em"
    )

    # Training losses
    results["plots"]["leduc_loss"]   = plot_training_loss(r2["training_log"], "Deep CFR: Leduc Hold'em Training Loss")
    results["plots"]["auction_loss"] = plot_training_loss(r3["training_log"], "Deep CFR: Auction Game Training Loss")

    # Payoff bars
    results["plots"]["leduc_payoff"] = plot_payoff_bars(
        r2["sim_vs_random"], ["Deep CFR", "Random"], "Leduc: Deep CFR vs Random"
    )
    results["plots"]["auction_payoff_eq"] = plot_payoff_bars(
        r3["sim_all_cfr"], ["Seller", "Buyer A", "Buyer B"], "Auction: Equilibrium Payoffs"
    )
    results["plots"]["auction_payoff_cfr_vs_rand"] = plot_payoff_bars(
        r3["sim_cfr_vs_random"], ["Seller (CFR)", "Buyer A (Random)", "Buyer B (Random)"],
        "Auction: CFR Seller vs Random Buyers"
    )

    print("  All plots generated.")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fast demo run")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    results = run_all(quick=args.quick)
    
    # Save generated plots directly to disk
    if "plots" in results:
        import base64
        for plot_name, b64_str in results["plots"].items():
            if b64_str:
                with open(f"{plot_name}.png", "wb") as img_file:
                    img_file.write(base64.b64decode(b64_str))
                print(f"Saved plot image: {plot_name}.png")

    # Remove plot data from JSON (too large), just save metrics
    save = {k: v for k, v in results.items() if k != "plots"}
    with open(args.output, "w") as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")
