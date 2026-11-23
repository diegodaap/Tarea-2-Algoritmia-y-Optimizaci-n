import time, os, numpy as np
import multiprocessing as mp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _count(args):
    n, seed = args
    rng = np.random.default_rng(seed)
    hits, resto, B = 0, n, 5_000_000      # por bloques para no agotar memoria
    while resto > 0:
        m = min(B, resto)
        x = rng.random(m); y = rng.random(m)
        hits += int(np.count_nonzero(x * x + y * y <= 1.0)); resto -= m
    return hits

def estimate_pi(n_samples, n_procs, seed=0):
    """Monte Carlo: reparte n_samples entre n_procs procesos."""
    chunk = n_samples // n_procs
    jobs = [(chunk, seed + i) for i in range(n_procs)]
    if n_procs == 1:
        hits = _count(jobs[0])
    else:
        with mp.Pool(n_procs) as p:
            hits = sum(p.map(_count, jobs))
    return 4.0 * hits / (chunk * n_procs)

if __name__ == "__main__":
    N = 200_000_000; reps = 3; procs = [1, 2, 4, 8]
    med = {}
    for p in procs:
        ts = []
        for r in range(reps):
            t0 = time.perf_counter(); pi = estimate_pi(N, p, seed=r * 100); ts.append(time.perf_counter() - t0)
        med[p] = float(np.median(ts))
        print(f"procs={p} tiempo={med[p]:.3f}s pi~{pi:.6f}")
    sp = [med[1] / med[p] for p in procs]
    print("speedup:", [round(s, 2) for s in sp], "| nucleos:", os.cpu_count())
    plt.figure(figsize=(6, 4.5))
    plt.plot(procs, sp, "o-", label="Speedup medido")
    plt.plot(procs, procs, "k--", label="Speedup ideal")
    plt.xlabel("Numero de procesos"); plt.ylabel("Speedup T1/Tp"); plt.xticks(procs)
    plt.title(f"estimate_pi (N={N:,}, mediana de {reps} corridas)"); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig("results/ej1_speedup.png", dpi=150)
