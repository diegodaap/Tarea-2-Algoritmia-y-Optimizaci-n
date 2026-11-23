import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, random

def greedy(items, cap, key):
    """items: lista (nombre, peso, valor). key ordena de mejor a peor."""
    sel, w, v = [], 0, 0
    for name, wi, vi in sorted(items, key=key):
        if w + wi <= cap:
            sel.append(name); w += wi; v += vi
    return sel, w, v

by_ratio = lambda it: -it[2] / it[1]
by_weight = lambda it: it[1]

def compare(items, cap, title, fname):
    r = greedy(items, cap, by_ratio); m = greedy(items, cap, by_weight)
    print(f"\n{title} (capacidad={cap})")
    print(f"{'Metodo':<22}{'Seleccion':<45}{'Peso':>6}{'Valor':>7}")
    print(f"{'Valor/peso':<22}{str(r[0]):<45}{r[1]:>6}{r[2]:>7}")
    print(f"{'Menor peso primero':<22}{str(m[0]):<45}{m[1]:>6}{m[2]:>7}")
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    lab = ["Valor/peso", "Menor peso"]
    ax[0].bar(lab, [r[2], m[2]], color=["tab:blue", "tab:orange"]); ax[0].set_title("Valor total")
    ax[1].bar(lab, [r[1], m[1]], color=["tab:blue", "tab:orange"]); ax[1].axhline(cap, color="r", ls="--", label="Capacidad"); ax[1].set_title("Capacidad usada"); ax[1].legend()
    for a, vals in zip(ax, [[r[2], m[2]], [r[1], m[1]]]):
        for i, x in enumerate(vals): a.text(i, x, str(x), ha="center", va="bottom")
    fig.suptitle(f"{title}\nValor/peso elige {r[0]} | Menor peso elige {m[0]}", fontsize=9)
    fig.tight_layout(); fig.savefig(fname, dpi=150)

# Contraejemplo: menor peso toma objetos ligeros pero poco valiosos y bloquea el objeto valioso
contra = [("A", 2, 1), ("B", 3, 2), ("C", 9, 18)]
compare(contra, 10, "Contraejemplo (menor peso es peor)", "results/ej2_contraejemplo.png")

rng = random.Random(7)
items = [(f"o{i}", rng.randint(2, 20), rng.randint(5, 60)) for i in range(10)]
print("\nObjetos (nombre, peso, valor):", items)
compare(items, 40, "Instancia aleatoria de 10 objetos", "results/ej2_aleatoria.png")
