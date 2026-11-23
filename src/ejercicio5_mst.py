import random, numpy as np, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tsp import *

INST, N = 100, 9
rng = random.Random(5)
R = {k: [] for k in ["MST", "MST + 2-opt", "2-opt (inicio aleatorio)", "Recocido Simulado"]}
opt = []
for _ in range(INST):
    _, D = random_instance(N, rng)
    o = held_karp(D); opt.append(o)
    t, c = mst_approx(D); R["MST"].append(c)
    R["MST + 2-opt"].append(two_opt(D, t)[1])
    r = list(range(N)); rng.shuffle(r); R["2-opt (inicio aleatorio)"].append(two_opt(D, r)[1])
    me = np.mean([D[i][j] for i in range(N) for j in range(i)])
    R["Recocido Simulado"].append(simulated_annealing(D, "exponencial", 20_000, me, me * 1e-3, rng)[1])
ratios = {k: np.array(v) / np.array(opt) for k, v in R.items()}
for k, v in ratios.items():
    print(f"{k:<26} razon media={v.mean():.4f} max={v.max():.4f} min={v.min():.4f} | % instancias optimas={100*np.mean(v < 1+1e-9):.0f}%")
with open("results/ej5_razones.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["instancia", "optimo"] + list(R))
    for i in range(INST): w.writerow([i, opt[i]] + [R[k][i] for k in R])

m = ratios["MST"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(m, bins=20, color="tab:blue", edgecolor="k"); ax[0].axvline(2, color="r", ls="--", label="Limite teorico = 2")
ax[0].axvline(m.mean(), color="g", ls=":", label=f"Media = {m.mean():.3f}")
ax[0].set_xlim(1, 2.1); ax[0].set_xlabel("C_obtenido / C_optimo"); ax[0].set_ylabel("Instancias"); ax[0].set_title("Razon de aproximacion del MST (100 instancias)"); ax[0].legend()
ax[1].boxplot(list(ratios.values()), tick_labels=["MST", "MST+2opt", "2-opt\naleat.", "SA"]); ax[1].axhline(1, color="k", ls=":")
ax[1].set_ylabel("C_obtenido / C_optimo"); ax[1].set_title("Comparacion de metodos (n=9)")
for a in ax: a.grid(alpha=.3)
fig.tight_layout(); fig.savefig("results/ej5_aproximacion.png", dpi=150)
