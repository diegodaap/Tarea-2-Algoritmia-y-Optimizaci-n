import random, time, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tsp import *

N_CITIES, RUNS = 30, 5
SCHED = ["lineal", "exponencial", "logaritmico"]
rng = random.Random(3)
pts, D = random_instance(N_CITIES, rng)
mean_edge = np.mean([D[i][j] for i in range(N_CITIES) for j in range(i)])
T0, Tf = mean_edge, mean_edge * 1e-3

# 1) Curvas de convergencia (N fijo)
N = 100_000
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for s in SCHED:
    h = np.array([simulated_annealing(D, s, N, T0, Tf, random.Random(100 + r))[2] for r in range(RUNS)])
    it = h[0, :, 0]; cur = h[:, :, 2].mean(0); T = h[0, :, 1]
    ax[0].plot(it, cur, label=s); ax[1].plot(T, cur, label=s)
ax[0].set_xlabel("Iteracion"); ax[0].set_ylabel(f"Costo actual (media de {RUNS} corridas)"); ax[0].set_title("Costo vs iteraciones")
ax[1].set_xscale("log"); ax[1].invert_xaxis(); ax[1].set_xlabel("Temperatura"); ax[1].set_title("Costo vs temperatura")
for a in ax: a.grid(alpha=.3); a.legend()
fig.tight_layout(); fig.savefig("results/ej4_convergencia.png", dpi=150)

# 2) Tiempo vs calidad (variando N)
plt.figure(figsize=(6.5, 4.5))
for s in SCHED:
    xs, ys = [], []
    for n_it in [10_000, 30_000, 100_000, 300_000]:
        ts, cs = [], []
        for r in range(RUNS):
            t0 = time.perf_counter()
            _, c, _ = simulated_annealing(D, s, n_it, T0, Tf, random.Random(200 + r))
            ts.append(time.perf_counter() - t0); cs.append(c)
        xs.append(np.mean(ts)); ys.append(np.mean(cs))
        print(f"{s:<12} N={n_it:>7} tiempo={np.mean(ts):.2f}s costo medio={np.mean(cs):.4f} (min {np.min(cs):.4f})")
    plt.plot(xs, ys, "o-", label=s)
plt.xscale("log"); plt.xlabel("Tiempo de ejecucion (s)"); plt.ylabel("Costo final medio (menor = mejor)")
plt.title("SA: tiempo vs calidad (puntos: 10k, 30k, 100k, 300k iter.)", fontsize=10); plt.grid(alpha=.3); plt.legend()
plt.tight_layout(); plt.savefig("results/ej4_tiempo_vs_calidad.png", dpi=150)
