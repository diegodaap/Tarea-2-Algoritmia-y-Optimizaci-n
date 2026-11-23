"""Utilidades comunes para TSP: instancias, 2-opt, Vecino Mas Cercano, Recocido Simulado, MST y Held-Karp."""
import math, random
import numpy as np


def random_instance(n, rng):
    pts = np.array([[rng.random(), rng.random()] for _ in range(n)])
    D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    return pts, D.tolist()


def tour_cost(D, t):
    return sum(D[t[i]][t[(i + 1) % len(t)]] for i in range(len(t)))


def nearest_neighbor(D, start):
    n = len(D)
    tour, seen = [start], {start}
    while len(tour) < n:
        c = tour[-1]
        nxt = min((j for j in range(n) if j not in seen), key=lambda j: D[c][j])
        tour.append(nxt); seen.add(nxt)
    return tour


def two_opt(D, tour):
    """2-opt (primera mejora). Devuelve (tour, costo, numero_de_mejoras)."""
    n = len(tour); t = list(tour); mejoras = 0; improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
                if D[a][c] + D[b][d] - D[a][b] - D[c][d] < -1e-12:
                    t[i + 1:j + 1] = reversed(t[i + 1:j + 1])
                    mejoras += 1; improved = True
    return t, tour_cost(D, t), mejoras


def temperature(schedule, k, N, T0, Tf):
    f = k / N
    if schedule == "lineal":
        return T0 - (T0 - Tf) * f
    if schedule == "exponencial":
        return T0 * (Tf / T0) ** f
    if schedule == "logaritmico":
        # T_k = T0 / (1 + ln(1+k)): enfria tan lento que no llega a T baja en tiempo finito
        return T0 / (1 + math.log(1 + k))
    raise ValueError(schedule)


def simulated_annealing(D, schedule, N, T0, Tf, rng, n_hist=200):
    """SA con movimiento 2-opt aleatorio desde solucion inicial aleatoria."""
    n = len(D)
    t = list(range(n)); rng.shuffle(t)
    cost = tour_cost(D, t); best = cost; best_t = t[:]
    hist = []
    step = max(1, N // n_hist)
    for k in range(N):
        T = temperature(schedule, k, N, T0, Tf)
        i = rng.randrange(n - 2)
        j = rng.randrange(i + 2, n)
        if not (i == 0 and j == n - 1):
            a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
            delta = D[a][c] + D[b][d] - D[a][b] - D[c][d]
            if delta < 0 or rng.random() < math.exp(-delta / T):
                t[i + 1:j + 1] = reversed(t[i + 1:j + 1])
                cost += delta
                if cost < best - 1e-12:
                    best = cost; best_t = t[:]
        if k % step == 0:
            hist.append((k, T, cost, best))
    return best_t, best, hist


def mst_approx(D):
    """2-aproximacion: MST (Prim) + recorrido en preorden."""
    n = len(D)
    in_t = [False] * n; key = [math.inf] * n; parent = [-1] * n; key[0] = 0
    children = [[] for _ in range(n)]
    for _ in range(n):
        u = min((v for v in range(n) if not in_t[v]), key=lambda v: key[v])
        in_t[u] = True
        if parent[u] >= 0:
            children[parent[u]].append(u)
        for v in range(n):
            if not in_t[v] and D[u][v] < key[v]:
                key[v] = D[u][v]; parent[v] = u
    tour = []
    def dfs(u):
        tour.append(u)
        for v in children[u]:
            dfs(v)
    dfs(0)
    return tour, tour_cost(D, tour)


def held_karp(D):
    """Optimo exacto (programacion dinamica), valido para n pequeno (n=9)."""
    n = len(D); INF = math.inf
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0
    for mask in range(1 << n):
        if not mask & 1:
            continue
        for last in range(n):
            if dp[mask][last] == INF:
                continue
            for nx in range(n):
                if mask & (1 << nx):
                    continue
                m2 = mask | (1 << nx)
                v = dp[mask][last] + D[last][nx]
                if v < dp[m2][nx]:
                    dp[m2][nx] = v
    full = (1 << n) - 1
    return min(dp[full][l] + D[l][0] for l in range(1, n))
