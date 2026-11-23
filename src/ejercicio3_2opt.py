import random, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tsp import *

N_CITIES, TRIALS = 30, 100
rng = random.Random(1)
res = {"Aleatoria": ([], [], []), "Vecino mas cercano": ([], [], [])}
for _ in range(TRIALS):
    _, D = random_instance(N_CITIES, rng)   # misma instancia para ambos inicios
    t = list(range(N_CITIES)); rng.shuffle(t)
    _, c, m = two_opt(D, t)
    res["Aleatoria"][0].append(m); res["Aleatoria"][1].append(c); res["Aleatoria"][2].append(tour_cost(D, t))
    nn = nearest_neighbor(D, rng.randrange(N_CITIES))
    _, c, m = two_opt(D, nn)
    res["Vecino mas cercano"][0].append(m); res["Vecino mas cercano"][1].append(c); res["Vecino mas cercano"][2].append(tour_cost(D, nn))
for k, (m, c, c0) in res.items():
    print(f"{k:<20} mejoras: media={np.mean(m):.1f} mediana={np.median(m):.0f} | costo inicial medio={np.mean(c0):.3f} | costo final medio={np.mean(c):.3f}")
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].boxplot([v[0] for v in res.values()], tick_labels=list(res)); ax[0].set_title("Mejoras hasta optimo local"); ax[0].set_ylabel("# mejoras")
ax[1].boxplot([v[1] for v in res.values()], tick_labels=list(res)); ax[1].set_title("Costo del optimo local"); ax[1].set_ylabel("Costo del tour")
for a in ax: a.grid(alpha=.3)
fig.suptitle(f"2-opt, {N_CITIES} ciudades, {TRIALS} instancias"); fig.tight_layout(); fig.savefig("results/ej3_2opt.png", dpi=150)
