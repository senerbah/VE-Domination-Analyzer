import networkx as nx
import time
import sys
sys.path.insert(0, '.')
from main import VEDominationAnalyzer

header = (
    f"{'Graph':<8}"
    f"{'gamma':>7}"
    f"{'Gamma':>7}"
    f"{'gap':>5}"
    f"{'BT calls':>14}"
    f"{'BT pruned':>12}"
    f"{'P1':>10}"
    f"{'P2':>10}"
    f"{'P3':>10}"
    f"{'time':>10}"
)
print(header, flush=True)
print("-" * 95, flush=True)

for n in [20, 30, 40, 50]:
    G = nx.cycle_graph(n)
    a = VEDominationAnalyzer(G)
    t0 = time.time()
    result = a.analyze(method='backtrack')
    elapsed = time.time() - t0
    st = result['stats']
    line = (
        f"C{n:<7}"
        f"{result['gamma']:>7}"
        f"{result['Gamma']:>7}"
        f"{result['gap']:>5}"
        f"{st['calls']:>14}"
        f"{st['pruned']:>12}"
        f"{st['pruned_zero_gain']:>10}"
        f"{st['pruned_private']:>10}"
        f"{st['pruned_dead_end']:>10}"
        f"{elapsed:>9.4f}s"
    )
    print(line, flush=True)
    print(f"  minimal sets: {len(result['minimal_sets'])}", flush=True)
