"""
Optional: calibrate decay_rate and noise_std via gradient descent, instead
of hand-picking them by manual sweeping (as Phase 2 originally did).

Why finite-difference gradients, not backprop: the recall step uses
argsort/top-k to enforce sparsity, which is a discrete operation with no
useful analytic gradient. Finite-difference gradient descent sidesteps this
by numerically estimating how the loss changes when each parameter is
nudged slightly -- this is real gradient descent, just with numerically
estimated gradients instead of analytically computed ones.

To keep the numerical gradient estimate stable despite the simulation being
stochastic, we use "common random numbers": the +/- perturbation runs use
the SAME per-trial random seeds, so random noise mostly cancels out of the
difference, leaving a much cleaner gradient signal.
"""

import numpy as np
from hebbian_sparse import SparseHebbianMemory


def simulate_accuracy_curve(decay_rate, noise_std, delay_steps, trials, base_seed,
                             n=200, sparsity=0.05, n_distractors=0, reveal_fraction=0.4):
    results = {}
    for delay in delay_steps:
        accs = []
        for t in range(trials):
            trial_seed = base_seed * 100_000 + t
            mem = SparseHebbianMemory(n, sparsity=sparsity, seed=trial_seed)
            pattern = mem.random_pattern()
            active_indices = np.where(pattern == 1)[0]
            mem.encode(pattern)
            for _ in range(delay):
                mem.decay(decay_rate, noise_std=noise_std)
            for _ in range(n_distractors):
                mem.encode(mem.random_pattern())
            cue, known_mask = mem.make_cue(active_indices, reveal_fraction)
            recalled = mem.recall(cue, known_mask)
            accs.append(mem.overlap_accuracy(recalled, pattern, mem.k))
        results[delay] = float(np.mean(accs))
    return results


def loss_fn(decay_rate, noise_std, target, delay_steps, trials, base_seed):
    sim = simulate_accuracy_curve(decay_rate, noise_std, delay_steps, trials, base_seed)
    return float(np.mean([(sim[d] - target[d]) ** 2 for d in delay_steps]))


def gradient_descent_calibrate(target, delay_steps=(0, 4, 8, 12, 16, 20),
                                trials=25, iterations=25, lr_decay=0.02, lr_noise=0.05,
                                eps=1e-3, decay_rate0=0.06, noise_std0=0.5, seed=0):
    decay_rate, noise_std = decay_rate0, noise_std0
    history = []

    for it in range(iterations):
        base_seed = seed * 1000 + it

        loss0 = loss_fn(decay_rate, noise_std, target, delay_steps, trials, base_seed)

        loss_dr_plus = loss_fn(decay_rate + eps, noise_std, target, delay_steps, trials, base_seed)
        grad_decay = (loss_dr_plus - loss0) / eps

        loss_ns_plus = loss_fn(decay_rate, noise_std + eps, target, delay_steps, trials, base_seed)
        grad_noise = (loss_ns_plus - loss0) / eps

        decay_rate = max(0.0, decay_rate - lr_decay * grad_decay)
        noise_std = max(0.0, noise_std - lr_noise * grad_noise)

        history.append((it, loss0, decay_rate, noise_std))
        print(f"iter {it:>2}  loss={loss0:.5f}  decay_rate={decay_rate:.4f}  noise_std={noise_std:.4f}")

    return decay_rate, noise_std, history


if __name__ == "__main__":
    delay_steps = (0, 4, 8, 12, 16, 20)
    target = {d: 0.4 + 0.6 * np.exp(-d / 8.0) for d in delay_steps}
    print("Target curve:", {d: round(v, 3) for d, v in target.items()})
    print()

    final_decay, final_noise, hist = gradient_descent_calibrate(target, delay_steps=delay_steps)

    print()
    print(f"Calibrated: decay_rate={final_decay:.4f}, noise_std={final_noise:.4f}")
    print()
    print("Final simulated curve vs target:")
    final_sim = simulate_accuracy_curve(final_decay, final_noise, delay_steps, trials=100, base_seed=9999)
    for d in delay_steps:
        print(f"  delay={d:>2}  simulated={final_sim[d]:.3f}  target={target[d]:.3f}")