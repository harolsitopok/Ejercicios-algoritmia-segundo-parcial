import os, sys, time, json
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import wilcoxon

if os.name == "nt":
    os.system("")  
C = {"b": "\033[1m", "c": "\033[96m", "g": "\033[92m", "y": "\033[93m", "r": "\033[0m"}

def banner(text):
    line = "═" * 68
    print(f"\n{C['c']}{C['b']}╔{line}╗\n║ {text.ljust(67)}║\n╚{line}╝{C['r']}")

def table(headers, rows):
    cols = list(zip(headers, *rows))
    w = [max(len(str(x)) for x in col) + 2 for col in cols]
    sep = lambda l, m, r: l + m.join("─" * x for x in w) + r
    fmt = lambda cells, st="": "│" + "│".join(
        f"{st}{str(c).center(x)}{C['r'] if st else ''}" for c, x in zip(cells, w)) + "│"
    print(sep("┌", "┬", "┐"))
    print(fmt(headers, C["b"] + C["y"]))
    print(sep("├", "┼", "┤"))
    for r in rows:
        print(fmt(r))
    print(sep("└", "┴", "┘"))

def ok(msg):
    print(f"{C['g']}✔ {msg}{C['r']}")
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "results")
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

def random_instance(n, rng, size=100.0):
    pts = rng.random((n, 2)) * size
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    return pts, d


def tour_cost(tour, d):
    t = np.asarray(tour)
    return float(d[t, np.roll(t, -1)].sum())


def nearest_neighbor(d, start):
    n = len(d)
    visited = np.zeros(n, bool)
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        row = np.where(visited, np.inf, d[tour[-1]])
        nxt = int(np.argmin(row))
        tour.append(nxt)
        visited[nxt] = True
    return tour


def two_opt(tour, d, max_passes=10_000):
    """2-opt con primera mejora. Devuelve (tour, n_mejoras, historial_costos)."""
    tour = list(tour)
    n = len(tour)
    cost = tour_cost(tour, d)
    history = [cost]
    improvements = 0
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, e = tour[j], tour[(j + 1) % n]
                delta = d[a, c] + d[b, e] - d[a, b] - d[c, e]
                if delta < -1e-12:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    cost += delta
                    improvements += 1
                    history.append(cost)
                    improved = True
    return tour, improvements, history


def optimal_tour_cost(d):
    """Held-Karp exacto (valido para n <= ~12)."""
    n = len(d)
    INF = float("inf")
    dp = {(1, 0): 0.0}
    for mask in range(1, 1 << n):
        if not mask & 1:
            continue
        for last in range(n):
            if (mask, last) not in dp:
                continue
            base = dp[(mask, last)]
            for nxt in range(n):
                if (mask >> nxt) & 1:
                    continue
                key = (mask | (1 << nxt), nxt)
                v = base + d[last, nxt]
                if v < dp.get(key, INF):
                    dp[key] = v
    full = (1 << n) - 1
    return min(dp[(full, last)] + d[last, 0] for last in range(1, n))


def mst_tour(d):
    n = len(d)
    in_tree = np.zeros(n, bool)
    in_tree[0] = True
    best = d[0].copy()
    parent = np.zeros(n, int)
    children = [[] for _ in range(n)]
    for _ in range(n - 1):
        cand = np.where(in_tree, np.inf, best)
        v = int(np.argmin(cand))
        children[parent[v]].append(v)
        in_tree[v] = True
        upd = (d[v] < best) & ~in_tree
        best[upd] = d[v][upd]
        parent[upd] = v
    tour, stack = [], [0]
    while stack:
        u = stack.pop()
        tour.append(u)
        stack.extend(reversed(children[u]))
    return tour


def simulated_annealing(d, init_tour, n_iter, schedule, T0, Tf, rng, record_every=50):
    n = len(d)
    tour = list(init_tour)
    cost = tour_cost(tour, d)
    best_cost, best_tour = cost, tour[:]
    c = (T0 / Tf - 1) / np.log(1 + n_iter) if schedule == "logarithmic" else 0.0
    hist_it, hist_cost, hist_T = [], [], []
    for k in range(n_iter):
        frac = k / n_iter
        if schedule == "linear":
            T = T0 - (T0 - Tf) * frac
        elif schedule == "exponential":
            T = T0 * (Tf / T0) ** frac
        else:
            T = T0 / (1 + c * np.log(1 + k))
        i, j = sorted(rng.choice(n, 2, replace=False))
        if j - i >= 2 and not (i == 0 and j == n - 1):
            a, b = tour[i], tour[i + 1]
            cc, e = tour[j], tour[(j + 1) % n]
            delta = d[a, cc] + d[b, e] - d[a, b] - d[cc, e]
            if delta < 0 or rng.random() < np.exp(-delta / T):
                tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                cost += delta
                if cost < best_cost - 1e-12:
                    best_cost, best_tour = cost, tour[:]
        if k % record_every == 0:
            hist_it.append(k); hist_cost.append(cost); hist_T.append(T)
    return best_tour, best_cost, (hist_it, hist_cost, hist_T)


# ======================================================================
# EJERCICIO 1
# ======================================================================
N_TOTAL = 40_000_000   # ej. 1
CHUNK = 1_000_000


def _count_inside(args):
    n, seed = args
    rng = np.random.default_rng(seed)
    inside, done = 0, 0
    while done < n:
        m = min(CHUNK, n - done)
        xy = rng.random((m, 2))
        inside += int(((xy ** 2).sum(1) <= 1.0).sum())
        done += m
    return inside


def estimate_pi(n_total, n_procs, seed=0):
    parts = [n_total // n_procs] * n_procs
    parts[0] += n_total - sum(parts)
    jobs = [(p, seed + i) for i, p in enumerate(parts)]
    if n_procs == 1:
        inside = _count_inside(jobs[0])
    else:
        with Pool(n_procs) as pool:
            inside = sum(pool.map(_count_inside, jobs))
    return 4.0 * inside / n_total


def amdahl(p, f):
    return 1.0 / (f + (1 - f) / p)


def run_ex1():
    procs = [1, 2, 4, 8]
    reps = 5
    times = {}
    for p in procs:
        ts = []
        for r in range(reps):
            t0 = time.perf_counter()
            pi = estimate_pi(N_TOTAL, p, seed=r * 100)
            ts.append(time.perf_counter() - t0)
        times[p] = float(np.median(ts))
    speed = [times[1] / times[p] for p in procs]
    eff = [s / p for s, p in zip(speed, procs)]
    table(["Procesos", "Tiempo (s)", "Speedup", "Eficiencia"],
        [[p, f"{times[p]:.3f}", f"{s:.2f}x", f"{e*100:.0f}%"]
         for p, s, e in zip(procs, speed, eff)])
    cores = os.cpu_count()
    try:
        popt, _ = curve_fit(amdahl, procs, speed, p0=[0.1], bounds=(0, 1))
        f_fit = float(popt[0])
    except Exception:
        f_fit = None
    json.dump({"cores": cores, "procs": procs, "times": times, "speedup": speed,
               "efficiency": eff, "amdahl_f_fit": f_fit},
              open(os.path.join(OUT, "ex1_results.json"), "w"), indent=2)
    ps = np.linspace(1, 8, 100)
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(procs, procs, "k--", label="Speedup ideal (lineal)")
    plt.plot(procs, speed, "o-", color="tab:blue", label="Speedup medido")
    if f_fit is not None:
        plt.plot(ps, amdahl(ps, f_fit), ":", color="tab:red", label=f"Ajuste Amdahl (f={f_fit:.2f})")
    plt.xlabel("Número de procesos"); plt.ylabel("Speedup  T1 / Tp")
    plt.title(f"Ej. 1: aceleración de estimate_pi (CPU: {cores} núcleo(s))")
    plt.xticks(procs); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figures", "ex1_speedup.png"), dpi=150)


# ======================================================================
# EJERCICIO 2
# ======================================================================
def greedy_knapsack(items, capacity, key):
    chosen, used, value = [], 0, 0
    for name, w, v in sorted(items, key=key):
        if used + w <= capacity:
            chosen.append(name); used += w; value += v
    return chosen, used, value


by_ratio = lambda it: -(it[2] / it[1])   # mayor valor/peso primero (original)
by_weight = lambda it: (it[1], -it[2])   # menor peso primero (modificado)


def compare(items, capacity):
    r = {}
    for label, key in (("valor/peso", by_ratio), ("menor peso", by_weight)):
        ch, u, v = greedy_knapsack(items, capacity, key)
        r[label] = {"objetos": ch, "capacidad_usada": u, "valor_total": v}
    return r


def bars(res, capacity, title, fname, ax_pair=None):
    labels = list(res)
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
    cols = ["tab:blue", "tab:orange"]
    ax[0].bar(labels, [len(res[l]["objetos"]) for l in labels], color=cols)
    ax[0].set_title("Objetos seleccionados")
    ax[1].bar(labels, [res[l]["capacidad_usada"] for l in labels], color=cols)
    ax[1].axhline(capacity, color="k", ls="--", label=f"Capacidad={capacity}")
    ax[1].set_title("Capacidad usada"); ax[1].legend(fontsize=8)
    ax[2].bar(labels, [res[l]["valor_total"] for l in labels], color=cols)
    ax[2].set_title("Valor total")
    for a in ax:
        a.grid(alpha=.3, axis="y")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", fname), dpi=150)
    plt.close(fig)


def run_ex2():
    # --- Contraejemplo pequeño: menor peso es peor que valor/peso ---
    cap = 8
    counter = [("A", 3, 1), ("B", 4, 2), ("C", 5, 30)]
    res_c = compare(counter, cap)
    
    bars(res_c, cap, "Contraejemplo (capacidad 8): A(3,1) B(4,2) C(5,30)", "ex2_contraejemplo.png")

    # --- Instancia aleatoria (20 objetos) ---
    rng = np.random.default_rng(7)
    items = [(f"o{i}", int(rng.integers(1, 20)), int(rng.integers(1, 60))) for i in range(20)]
    cap2 = 60
    res_r = compare(items, cap2)
    
    bars(res_r, cap2, "Instancia aleatoria (20 objetos, capacidad 60)", "ex2_aleatoria.png")

    # --- Tabla CSV y resumen sobre 500 instancias ---
    wins = {"valor/peso": 0, "menor peso": 0, "empate": 0}
    for s in range(500):
        r = np.random.default_rng(s)
        its = [(f"o{i}", int(r.integers(1, 20)), int(r.integers(1, 60))) for i in range(20)]
        rs = compare(its, 60)
        a, b = rs["valor/peso"]["valor_total"], rs["menor peso"]["valor_total"]
        wins["valor/peso" if a > b else "menor peso" if b > a else "empate"] += 1
        
    json.dump({"contraejemplo": {"items": counter, "capacidad": cap, "resultado": res_c},
               "aleatoria": {"items": items, "capacidad": cap2, "resultado": res_r},
               "victorias_500": wins}, open(os.path.join(OUT, "ex2_results.json"), "w"), indent=2)
    
    fila = lambda n, r: [n, ", ".join(r["objetos"]), r["capacidad_usada"], r["valor_total"]]
    H = ["Método", "Objetos", "Capacidad usada", "Valor total"]
    print("Contraejemplo (capacidad 8)")
    table(H, [fila(k, v) for k, v in res_c.items()])
    print("Instancia aleatoria (capacidad 60)")
    table(H, [fila(k, v) for k, v in res_r.items()])
    print("Victorias en 500 instancias")
    table(["Método", "Victorias"], [[k, v] for k, v in wins.items()])


# ======================================================================
# EJERCICIO 3
# ======================================================================


def run_ex3():
    N, INSTANCES = 30, 100
    rng = np.random.default_rng(42)
    rows = []
    for k in range(INSTANCES):
        _, d = random_instance(N, rng)
        t_rand = list(rng.permutation(N))
        t_nn = nearest_neighbor(d, int(rng.integers(N)))
        c0_r, c0_n = tour_cost(t_rand, d), tour_cost(t_nn, d)
        _, imp_r, _ = two_opt(t_rand, d)
        t_r2, _, _ = two_opt(t_rand, d)
        t_n2, imp_n, _ = two_opt(t_nn, d)
        rows.append(dict(inst=k, init_rand=c0_r, init_nn=c0_n, imp_rand=imp_r, imp_nn=imp_n,
                         final_rand=tour_cost(t_r2, d), final_nn=tour_cost(t_n2, d)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "ex3_results.csv"), index=False)
    summ = {c: dict(mean=float(df[c].mean()), median=float(df[c].median()), std=float(df[c].std()))
            for c in df.columns if c != "inst"}
    summ["wilcoxon_mejoras_p"] = float(wilcoxon(df.imp_rand, df.imp_nn).pvalue)
    summ["wilcoxon_calidad_p"] = float(wilcoxon(df.final_rand, df.final_nn).pvalue)
    summ["frac_nn_menos_mejoras"] = float((df.imp_nn < df.imp_rand).mean())
    summ["frac_nn_mejor_o_igual_calidad"] = float((df.final_nn <= df.final_rand).mean())
    json.dump(summ, open(os.path.join(OUT, "ex3_summary.json"), "w"), indent=2)
    
    table(["Inicio", "Mejoras (media)", "Costo final (media)"],
          [["Aleatorio", f"{summ['imp_rand']['mean']:.1f}", f"{summ['final_rand']['mean']:.1f}"],
           ["Vecino más cercano", f"{summ['imp_nn']['mean']:.1f}", f"{summ['final_nn']['mean']:.1f}"]])
    ok(f"NN usó menos mejoras en {summ['frac_nn_menos_mejoras']*100:.0f}% de las instancias "
       f"(Wilcoxon p={summ['wilcoxon_mejoras_p']:.1e})")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].boxplot([df.imp_rand, df.imp_nn], tick_labels=["Inicio aleatorio", "Vecino más cercano"])
    ax[0].set_ylabel("Nº de mejoras 2-opt"); ax[0].set_title("Mejoras hasta óptimo local")
    ax[1].boxplot([df.final_rand, df.final_nn], tick_labels=["Inicio aleatorio", "Vecino más cercano"])
    ax[1].set_ylabel("Costo del óptimo local"); ax[1].set_title("Calidad del óptimo local")
    for a in ax: a.grid(alpha=.3, axis="y")
    fig.suptitle(f"Ej. 3: 2-opt, {INSTANCES} instancias de {N} ciudades")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "ex3_boxplots.png"), dpi=150)


# ======================================================================
# EJERCICIO 4
# ======================================================================
SCHEDULES = ["linear", "exponential", "logarithmic"]
BUDGETS = [5_000, 20_000, 60_000]
RUNS = 10
COLORS = {"linear": "tab:blue", "exponential": "tab:green", "logarithmic": "tab:red"}


def run_ex4():
    N = 40
    T0, TF = 100.0, 0.1
    rng_inst = np.random.default_rng(2024)
    _, d = random_instance(N, rng_inst)
    results = {s: {b: {"cost": [], "time": []} for b in BUDGETS} for s in SCHEDULES}
    curves = {}
    for s in SCHEDULES:
        for b in BUDGETS:
            for r in range(RUNS):
                rng = np.random.default_rng(1000 * r + 7)
                init = list(rng.permutation(N))
                t0 = time.perf_counter()
                _, bc, h = simulated_annealing(d, init, b, s, T0, TF, rng, record_every=max(1, b // 400))
                results[s][b]["cost"].append(bc)
                results[s][b]["time"].append(time.perf_counter() - t0)
                if b == 20_000 and r == 0:
                    curves[s] = h

    # 1) Curvas de convergencia (presupuesto 20000)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for s in SCHEDULES:
        it, c, T = curves[s]
        ax[0].plot(it, c, color=COLORS[s], label=s, lw=1)
        ax[1].plot(T, c, color=COLORS[s], label=s, lw=1)
    ax[0].set_xlabel("Iteración"); ax[0].set_ylabel("Costo actual"); ax[0].set_title("Costo vs iteraciones")
    ax[1].set_xlabel("Temperatura (log)"); ax[1].set_xscale("log"); ax[1].invert_xaxis()
    ax[1].set_title("Costo vs temperatura")
    for a in ax: a.grid(alpha=.3); a.legend()
    fig.suptitle(f"Ej. 4: convergencia de SA (n={N}, 20 000 iteraciones)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "ex4_convergencia.png"), dpi=150)
    plt.close(fig)

    # 2) Tiempo vs calidad
    plt.figure(figsize=(7, 4.8))
    for s in SCHEDULES:
        xs = [np.mean(results[s][b]["time"]) for b in BUDGETS]
        ys = [np.mean(results[s][b]["cost"]) for b in BUDGETS]
        es = [np.std(results[s][b]["cost"]) for b in BUDGETS]
        plt.errorbar(xs, ys, yerr=es, fmt="o-", color=COLORS[s], capsize=3, label=s)
        for x, y, b in zip(xs, ys, BUDGETS):
            plt.annotate(f"{b//1000}k", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel("Tiempo de ejecución (s)"); plt.ylabel("Costo final (media ± desv. est.)")
    plt.title("Ej. 4: tiempo vs calidad (etiquetas = iteraciones)")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figures", "ex4_tiempo_vs_calidad.png"), dpi=150)

    out = {s: {str(b): {"cost_mean": float(np.mean(results[s][b]["cost"])),
                        "cost_std": float(np.std(results[s][b]["cost"])),
                        "cost_min": float(np.min(results[s][b]["cost"])),
                        "time_mean": float(np.mean(results[s][b]["time"]))} for b in BUDGETS}
           for s in SCHEDULES}
    
    rows = [[s, f"{b:,}", f"{out[s][str(b)]['cost_mean']:.1f} ± {out[s][str(b)]['cost_std']:.1f}",
             f"{out[s][str(b)]['time_mean']:.2f}"] for s in SCHEDULES for b in BUDGETS]
    table(["Esquema", "Iteraciones", "Costo final", "Tiempo (s)"], rows)
    json.dump(out, open(os.path.join(OUT, "ex4_results.json"), "w"), indent=2)


# ======================================================================
# EJERCICIO 5
# ======================================================================



def run_ex5():
    N, INSTANCES, SA_ITERS = 9, 100, 5_000
    rng = np.random.default_rng(99)
    rows = []
    for k in range(INSTANCES):
        _, d = random_instance(N, rng)
        opt = optimal_tour_cost(d)
        t_mst = mst_tour(d)
        c_mst = tour_cost(t_mst, d)
        t_m2, _, _ = two_opt(t_mst, d)
        t_ls, _, _ = two_opt(list(rng.permutation(N)), d)            # busqueda local (aleatorio)
        t_nn, _, _ = two_opt(nearest_neighbor(d, int(rng.integers(N))), d)
        _, c_sa, _ = simulated_annealing(d, list(rng.permutation(N)), SA_ITERS, "exponential",
                                         30.0, 0.1, rng, record_every=SA_ITERS)
        rows.append(dict(inst=k, optimo=opt, mst=c_mst, mst_2opt=tour_cost(t_m2, d),
                         ls_aleatorio=tour_cost(t_ls, d), ls_nn=tour_cost(t_nn, d), sa=c_sa))
    df = pd.DataFrame(rows)
    methods = ["mst", "mst_2opt", "ls_aleatorio", "ls_nn", "sa"]
    for m in methods:
        df["r_" + m] = df[m] / df["optimo"]
    df.to_csv(os.path.join(OUT, "ex5_results.csv"), index=False)
    summ = {m: dict(ratio_mean=float(df["r_" + m].mean()), ratio_median=float(df["r_" + m].median()),
                    ratio_max=float(df["r_" + m].max()), ratio_min=float(df["r_" + m].min()),
                    frac_optimo=float((df["r_" + m] < 1 + 1e-9).mean()),
                    cost_mean=float(df[m].mean())) for m in methods}
    summ["cota_2_violada"] = int((df.r_mst > 2 + 1e-9).sum())
    json.dump(summ, open(os.path.join(OUT, "ex5_summary.json"), "w"), indent=2)
    
    names = {"mst": "MST puro", "mst_2opt": "MST + 2-opt", "ls_aleatorio": "2-opt (aleatorio)",
             "ls_nn": "2-opt (NN)", "sa": "Recocido simulado"}
    table(["Método", "Razón media", "Razón máx.", "% óptimas", "Costo medio"],
          [[names[m], f"{summ[m]['ratio_mean']:.3f}", f"{summ[m]['ratio_max']:.3f}",
            f"{summ[m]['frac_optimo']*100:.0f}%", f"{summ[m]['cost_mean']:.1f}"] for m in methods])
    ok(f"Cota teórica de 2 violada en {summ['cota_2_violada']} de {INSTANCES} instancias")

    # 1) Razon de aproximacion
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(df.r_mst, bins=20, color="tab:blue", edgecolor="w")
    ax[0].axvline(2, color="red", ls="--", label="Límite teórico = 2")
    ax[0].axvline(df.r_mst.mean(), color="k", ls=":", label=f"Media = {df.r_mst.mean():.3f}")
    ax[0].set_xlim(0.95, 2.1); ax[0].set_xlabel("C_MST / C_óptimo"); ax[0].set_ylabel("Instancias")
    ax[0].set_title("Histograma de la razón de aproximación"); ax[0].legend()
    ax[1].scatter(range(INSTANCES), df.r_mst, s=14, color="tab:blue")
    ax[1].axhline(2, color="red", ls="--", label="Límite teórico = 2")
    ax[1].axhline(1, color="green", ls="--", label="Óptimo = 1")
    ax[1].set_ylim(0.9, 2.1); ax[1].set_xlabel("Instancia"); ax[1].set_ylabel("Razón")
    ax[1].set_title("Dispersión por instancia"); ax[1].legend()
    for a in ax: a.grid(alpha=.3)
    fig.suptitle("Ej. 5: razón de aproximación MST (100 instancias, 9 ciudades)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "ex5_razon_aproximacion.png"), dpi=150)
    plt.close(fig)

    # 2) Comparacion de costos
    labels = ["Óptimo", "MST", "MST+2-opt", "2-opt\n(aleatorio)", "2-opt\n(NN)", "Recocido\nsimulado"]
    cols = ["optimo"] + methods
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].boxplot([df[c] for c in cols], tick_labels=labels); ax[0].set_ylabel("Costo del tour")
    ax[0].set_title("Costo obtenido")
    ax[1].boxplot([df["r_" + m] for m in methods], tick_labels=labels[1:])
    ax[1].axhline(1, color="green", ls="--"); ax[1].set_ylabel("Razón C / C_óptimo")
    ax[1].set_title("Razón de aproximación por método")
    for a in ax: a.grid(alpha=.3, axis="y")
    fig.suptitle("Ej. 5: comparación entre métodos")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "ex5_comparacion_metodos.png"), dpi=150)


if __name__ == "__main__":
    runs = {1: run_ex1, 2: run_ex2, 3: run_ex3, 4: run_ex4, 5: run_ex5}
    titles = {1: "Ejercicio 1 · Aceleración paralela (Monte Carlo π)",
              2: "Ejercicio 2 · Mochila voraz (valor/peso vs menor peso)",
              3: "Ejercicio 3 · 2-opt: aleatorio vs vecino más cercano",
              4: "Ejercicio 4 · Recocido simulado: esquemas de enfriamiento",
              5: "Ejercicio 5 · Aproximación MST (100 instancias, 9 ciudades)"}
    chosen = [int(a) for a in sys.argv[1:]] or list(runs)
    for k in chosen:
        banner(titles[k])
        runs[k]()
    print(f"\n{C['g']}{C['b']}Listo. Gráficas en results/figures/{C['r']}")