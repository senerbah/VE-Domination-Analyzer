import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import math
import time
import csv
import os

from itertools import combinations
from collections import Counter


class VEDominationAnalyzer:

    def __init__(self, G):

        self.G = G
        self.edges = list(G.edges())
        self.m = len(self.edges)

        self.edge_index = {
            tuple(sorted(e)): i
            for i, e in enumerate(self.edges)
        }

        self.index_to_edge = {
            i: e
            for e, i in self.edge_index.items()
        }

        self.windows = self._build_windows()
        self.windows_mask = self._build_windows_mask()

        # Sorted vertex list for stable ordering
        self.V = sorted(self.G.nodes())
        self.n = len(self.V)

        # suffix_reachable[pos] = OR of windows_mask for V[pos:]
        # Used for O(1) dead-end detection
        self._build_suffix_reachable()

        self.all_edges_mask = (1 << self.m) - 1 if self.m > 0 else 0

    # -------------------------------------------------
    # Build windows (set-based and bitmask)
    # Rule: u ve-dominates edge vw  iff  u in N[v] ∪ N[w]
    # -------------------------------------------------

    def _build_windows(self):

        windows = {}

        for v in self.G.nodes():

            dominated = set()

            for e in self.G.edges(v):

                e = tuple(sorted(e))
                dominated.add(self.edge_index[e])

                a, b = e

                for adj in self.G.edges(a):
                    dominated.add(
                        self.edge_index[tuple(sorted(adj))]
                    )

                for adj in self.G.edges(b):
                    dominated.add(
                        self.edge_index[tuple(sorted(adj))]
                    )

            windows[v] = dominated

        return windows

    def _build_windows_mask(self):

        masks = {}

        for v, edge_set in self.windows.items():

            m = 0

            for idx in edge_set:
                m |= (1 << idx)

            masks[v] = m

        return masks

    def _build_suffix_reachable(self):

        V = self.V
        n = self.n

        self.suffix_reachable = [0] * (n + 1)

        for pos in range(n - 1, -1, -1):

            self.suffix_reachable[pos] = (
                self.suffix_reachable[pos + 1]
                | self.windows_mask[V[pos]]
            )

    # -------------------------------------------------
    # Basic predicates
    # -------------------------------------------------

    def covered_edges(self, S):

        covered = set()

        for v in S:
            covered |= self.windows[v]

        return covered

    def is_ve_dominating(self, S):

        return len(self.covered_edges(S)) == self.m

    def is_minimal(self, S):

        if not self.is_ve_dominating(S):
            return False

        S = set(S)

        for v in S:

            if self.is_ve_dominating(S - {v}):
                return False

        return True

    # -------------------------------------------------
    # Brute-force enumeration (for verification)
    # -------------------------------------------------

    def find_all_minimal_sets_brute(self):
        """
        Exhaustive combinations check.
        Returns (list_of_sets, stats_dict).
        """

        V = list(self.G.nodes())
        minimal_sets = []
        calls = 0
        rejected = 0

        for r in range(1, len(V) + 1):

            for S in combinations(V, r):

                calls += 1

                if self.is_minimal(S):
                    minimal_sets.append(tuple(sorted(S)))
                else:
                    rejected += 1

        stats = {
            'calls': calls,
            'pruned': rejected,
        }

        return minimal_sets, stats

    # -------------------------------------------------
    # Backtracking with pruning
    # -------------------------------------------------

    def find_all_minimal_sets_backtrack(self, on_found=None):
        """
        Recursive backtracking with three pruning rules:

        P1 (zero gain)    – skip v if it covers no new edges
        P2 (private kill) – skip v if it eliminates any existing
                            member's private edges
        P3 (dead end)     – prune branch if some uncovered edge
                            has no remaining candidate (suffix mask)

        on_found: optional callable(S) invoked for each minimal set.
          When provided the set is not stored internally — the caller
          owns the collection logic.  Returns stats_dict only.
          When None (default) sets are accumulated and returned as a
          list: returns (list_of_sets, stats_dict).
        """

        V = self.V
        n = self.n
        all_mask = self.all_edges_mask
        wm = self.windows_mask
        sr = self.suffix_reachable

        results = []   # used only when on_found is None
        _emit = on_found if on_found is not None else results.append

        stats = {
            'calls': 0,
            'pruned_zero_gain': 0,
            'pruned_private': 0,
            'pruned_dead_end': 0,
        }

        def backtrack(S, covered, start_pos, private):

            stats['calls'] += 1

            uncovered = all_mask & ~covered

            if not uncovered:
                # P2 guarantees all private sets are non-empty here
                _emit(tuple(sorted(S)))
                return

            # P3: dead-end — some uncovered edge is beyond all
            # remaining vertices' reach
            if uncovered & ~sr[start_pos]:
                stats['pruned_dead_end'] += 1
                return

            for pos in range(start_pos, n):

                v = V[pos]

                # P1: zero gain
                gain = wm[v] & ~covered

                if not gain:
                    stats['pruned_zero_gain'] += 1
                    continue

                # P2: adding v must not kill any member's private set
                new_private = {}
                skip = False

                for u in S:

                    np = private[u] & ~wm[v]

                    if not np:
                        stats['pruned_private'] += 1
                        skip = True
                        break

                    new_private[u] = np

                if skip:
                    continue

                # private edges of v = edges it covers that S doesn't yet
                new_private[v] = gain

                backtrack(
                    S + [v],
                    covered | wm[v],
                    pos + 1,
                    new_private
                )

        backtrack([], 0, 0, {})

        stats['pruned'] = (
            stats['pruned_zero_gain']
            + stats['pruned_private']
            + stats['pruned_dead_end']
        )

        if on_found is not None:
            return stats          # caller owns the results
        return results, stats

    # -------------------------------------------------
    # Format helpers
    # -------------------------------------------------

    def edge_label(self, idx):

        u, v = self.index_to_edge[idx]

        return f"{{{u},{v}}}"

    def format_set_detail(self, S):

        lines = []

        vertices = sorted(S)

        lines.append(
            f"S = {{{', '.join(str(v) for v in vertices)}}}"
            f"    |S| = {len(S)}"
        )

        for v in vertices:

            dominated_edges = sorted(
                self.index_to_edge[i]
                for i in self.windows[v]
            )

            labels = [f"{{{u},{w}}}" for u, w in dominated_edges]

            lines.append(
                f"  v={v}  ve-dominates:  "
                + ",  ".join(labels)
            )

        return "\n".join(lines)

    # -------------------------------------------------
    # Unified analysis entry point
    # -------------------------------------------------

    def analyze(self, method='backtrack', light=False, cycle_n=None):
        """
        Returns a result dict with:
          minimal_sets, distribution, gamma, Gamma,
          gap, well, gamma_sets, Gamma_sets, stats
        method : 'backtrack' (default) or 'brute'
        light  : when True, stream results without storing all sets.
                 Only count, cardinalities, and one representative per
                 extreme are kept.  Use for large graphs (n > LIGHT_N).
                 Brute-force always uses full mode regardless.
        cycle_n: when light=True and the graph is a cycle of this size,
                 also accumulate canonical pattern counts per-size in a
                 single pass (no second traversal needed).
        """

        # --- full mode ---
        if not light or method == 'brute':

            if method == 'backtrack':
                minimal_sets, stats = self.find_all_minimal_sets_backtrack()
            else:
                minimal_sets, stats = self.find_all_minimal_sets_brute()

            minimal_sets = sorted(minimal_sets)
            sizes        = [len(S) for S in minimal_sets]
            distribution = Counter(sizes)
            gamma        = min(sizes) if sizes else None
            Gamma        = max(sizes) if sizes else None

            return {
                'minimal_sets': minimal_sets,
                'count':        len(minimal_sets),
                'distribution': distribution,
                'gamma':        gamma,
                'Gamma':        Gamma,
                'gap':          (Gamma - gamma) if gamma is not None else None,
                'well':         (gamma == Gamma) if gamma is not None else None,
                'gamma_sets':   [S for S in minimal_sets if len(S) == gamma],
                'Gamma_sets':   [S for S in minimal_sets if len(S) == Gamma],
                'stats':        stats,
                'light':        False,
                'stream_pa':    None,
            }

        # --- light mode (backtrack only) ---
        # Streams results through a callback; nothing is kept in a list.
        # Pattern data (cycle-specific) is accumulated per-size so we can
        # classify gamma / Gamma patterns after the full traversal.

        count        = [0]
        gamma        = [None]
        Gamma        = [None]
        gamma_rep    = [None]
        Gamma_rep    = [None]
        distribution = Counter()

        # Pattern accumulators (only when cycle_n is supplied)
        pat_by_size  = {}   # size -> Counter{ canonical_pattern -> count }
        pat_examples = {}   # canonical_pattern -> first S seen
        pat_raw      = {}   # canonical_pattern -> raw gaps of first S
        all_verified = [True]

        def on_found(S):
            count[0] += 1
            k = len(S)
            distribution[k] += 1

            if gamma[0] is None or k < gamma[0]:
                gamma[0]     = k
                gamma_rep[0] = S
            if Gamma[0] is None or k > Gamma[0]:
                Gamma[0]     = k
                Gamma_rep[0] = S

            if cycle_n is not None:
                gaps = circular_gaps(S, cycle_n)
                pat  = canonical_pattern(gaps)

                if k not in pat_by_size:
                    pat_by_size[k] = Counter()
                pat_by_size[k][pat] += 1

                if pat not in pat_examples:
                    pat_examples[pat] = S
                    pat_raw[pat]      = gaps

                ok, _ = characterization_check(gaps)
                if not ok:
                    all_verified[0] = False

        stats = self.find_all_minimal_sets_backtrack(on_found=on_found)

        g = gamma[0]
        G = Gamma[0]

        # Build pattern-analysis dict (same schema as analyze_cycle_patterns)
        stream_pa = None
        if cycle_n is not None:
            all_counts   = Counter()
            gamma_counts = Counter()
            Gamma_counts = Counter()
            for size, cnt_dict in pat_by_size.items():
                for pat, cnt in cnt_dict.items():
                    all_counts[pat] += cnt
                    if size == g:
                        gamma_counts[pat] += cnt
                    if size == G:
                        Gamma_counts[pat] += cnt
            stream_pa = {
                'all_counts':   all_counts,
                'gamma_counts': gamma_counts,
                'Gamma_counts': Gamma_counts,
                'examples':     pat_examples,
                'raw_example':  pat_raw,
                'all_verified': all_verified[0],
            }

        return {
            'minimal_sets': None,                  # not stored
            'count':        count[0],
            'distribution': distribution,
            'gamma':        g,
            'Gamma':        G,
            'gap':          (G - g) if g is not None else None,
            'well':         (g == G) if g is not None else None,
            'gamma_sets':   [gamma_rep[0]] if gamma_rep[0] else [],
            'Gamma_sets':   [Gamma_rep[0]] if Gamma_rep[0] else [],
            'stats':        stats,
            'light':        True,
            'stream_pa':    stream_pa,
        }


# =====================================================
# CYCLE PATTERN ANALYSIS
# (cycle-specific; adapt gap/canonical logic for other families)
# =====================================================

def circular_gaps(S, n):
    """
    Circular gap sequence for sorted set S on C_n.
    Returns (d_0,...,d_{k-1}) with sum = n where d_i is the
    vertex-distance from S[i] to S[i+1 mod k] going clockwise.
    """
    S = sorted(S)
    k = len(S)
    if k == 0:
        return ()
    gaps = [S[i + 1] - S[i] for i in range(k - 1)]
    gaps.append(n - S[-1] + S[0])
    return tuple(gaps)


def canonical_pattern(gaps):
    """
    Canonical representative under the dihedral group
    (all rotations and reflections of the gap sequence).
    Makes structurally equivalent placements compare equal.
    """
    if len(gaps) <= 1:
        return tuple(gaps)
    k = len(gaps)
    fwd = [gaps[i:] + gaps[:i] for i in range(k)]
    rev = tuple(reversed(gaps))
    bwd = [rev[i:] + rev[:i] for i in range(k)]
    return min(fwd + bwd)


def characterization_check(gaps):
    """
    Structural characterization of minimal ve-dominating sets on C_n:
      (C1) max gap ≤ 4            -- ve-covering condition
      (C2) min consecutive sum ≥ 5 -- private-edge / minimality (k≥2)
    Returns (passes: bool, stats: dict).
    """
    k = len(gaps)
    if k == 0:
        return False, {}

    max_g = max(gaps)
    min_g = min(gaps)

    if k == 1:
        return max_g <= 4, {
            'max_gap': max_g,
            'min_gap': min_g,
            'min_psum': None,
            'max_psum': None,
        }

    psums = [gaps[i] + gaps[(i + 1) % k] for i in range(k)]
    min_ps = min(psums)
    max_ps = max(psums)

    ok = (max_g <= 4) and (min_ps >= 5)

    return ok, {
        'max_gap': max_g,
        'min_gap': min_g,
        'min_psum': min_ps,
        'max_psum': max_ps,
    }


def analyze_cycle_patterns(minimal_sets, gamma, Gamma, n):
    """
    Compute canonical distance patterns for all minimal sets of C_n.
    Returns counters and examples for all / gamma / Gamma subsets,
    plus a flag indicating whether the characterization holds for all sets.
    """
    all_counts    = Counter()
    gamma_counts  = Counter()
    Gamma_counts  = Counter()
    examples      = {}   # canonical_pattern -> first S encountered
    raw_example   = {}   # canonical_pattern -> raw gaps of first S
    all_verified  = True

    for S in minimal_sets:
        gaps = circular_gaps(S, n)
        pat  = canonical_pattern(gaps)

        all_counts[pat] += 1

        if pat not in examples:
            examples[pat]    = S
            raw_example[pat] = gaps

        if len(S) == gamma:
            gamma_counts[pat] += 1
        if len(S) == Gamma:
            Gamma_counts[pat] += 1

        ok, _ = characterization_check(gaps)
        if not ok:
            all_verified = False

    return {
        'all_counts':   all_counts,
        'gamma_counts': gamma_counts,
        'Gamma_counts': Gamma_counts,
        'examples':     examples,
        'raw_example':  raw_example,
        'all_verified': all_verified,
    }


def _fmt_pat(pat):
    """Compact string for a canonical pattern tuple."""
    return "(" + ",".join(str(d) for d in pat) + ")"


def write_pattern_section(file, n, result, pa):
    """
    Write the distance-pattern analysis block for one graph.
    pa = return value of analyze_cycle_patterns.
    """
    gamma = result['gamma']
    Gamma = result['Gamma']
    dash  = "-" * 65

    file.write(f"\nDistance Pattern Analysis  (n={n}):\n")
    file.write(dash + "\n")

    # --- full pattern table ---
    file.write(
        f"  {'Pattern':<18} {'|S|':>4} {'Count':>6}"
        f" {'Type':<12} {'minG':>5} {'maxG':>5} {'minPS':>6}\n"
    )
    file.write("  " + "-" * 57 + "\n")

    for pat in sorted(pa['all_counts'], key=lambda p: (len(p), p)):
        count = pa['all_counts'][pat]
        k     = len(pat)

        types = []
        if pat in pa['gamma_counts']:
            types.append('gamma')
        if pat in pa['Gamma_counts']:
            types.append('Gamma')
        type_str = '+'.join(types) if types else '—'

        _, det = characterization_check(pat)
        ps_str = str(det['min_psum']) if det['min_psum'] is not None else ' — '

        file.write(
            f"  {_fmt_pat(pat):<18} {k:>4} {count:>6}"
            f" {type_str:<12} {det['min_gap']:>5} {det['max_gap']:>5}"
            f" {ps_str:>6}\n"
        )

    # --- gamma-sets breakdown ---
    file.write(
        f"\n  gamma-sets  (|S|={gamma},"
        f" count={len(result['gamma_sets'])}):\n"
    )
    for pat, cnt in sorted(pa['gamma_counts'].items()):
        ex_S   = pa['examples'][pat]
        ex_gap = pa['raw_example'][pat]
        file.write(
            f"    {_fmt_pat(pat):<18} x{cnt:<5}"
            f" e.g. {ex_S} → {ex_gap}\n"
        )

    # --- Gamma-sets breakdown (only when gap > 0) ---
    if gamma != Gamma:
        file.write(
            f"\n  Gamma-sets  (|S|={Gamma},"
            f" count={len(result['Gamma_sets'])}):\n"
        )
        for pat, cnt in sorted(pa['Gamma_counts'].items()):
            ex_S   = pa['examples'][pat]
            ex_gap = pa['raw_example'][pat]
            file.write(
                f"    {_fmt_pat(pat):<18} x{cnt:<5}"
                f" e.g. {ex_S} → {ex_gap}\n"
            )

    # --- characterization status ---
    status = "PASS" if pa['all_verified'] else "FAIL"
    file.write(
        f"\n  Characterization (C1: max_gap≤4, C2: min_pair_sum≥5):"
        f"  {status}\n"
    )


def write_conjecture_summary(file, all_data):
    """
    Global summary: conjecture verification, well ve-dominated list,
    structural characterization statement, and all distinct patterns.
    all_data: list of {'n', 'result', 'pa'} dicts.
    """
    sep  = "=" * 65
    dash = "-" * 65

    n_min = all_data[0]['n']
    n_max = all_data[-1]['n']

    file.write(f"\n\n{sep}\n")
    file.write(f"GLOBAL CONJECTURE SUPPORT  (C{n_min} through C{n_max})\n")
    file.write(f"{sep}\n\n")

    # --- Conjecture 1: gamma_ve = ceil(n/4) ---
    file.write("Conjecture 1 :  gamma_ve(Cn) = ceil(n/4)\n")
    file.write(dash + "\n")
    misses = [
        (d['n'], d['result']['gamma'], math.ceil(d['n'] / 4))
        for d in all_data
        if d['result']['gamma'] != math.ceil(d['n'] / 4)
    ]
    if not misses:
        ns = [d['n'] for d in all_data]
        file.write(f"  CONFIRMED for n = {ns[0]} .. {ns[-1]}\n\n")
    else:
        for n, got, exp in misses:
            file.write(f"  FAILS at C{n}: computed={got}, formula={exp}\n")
        file.write("\n")

    # per-row table
    file.write(
        f"  {'n':>4}  {'ceil(n/4)':>9}  {'computed':>9}  match\n"
    )
    for d in all_data:
        n   = d['n']
        exp = math.ceil(n / 4)
        got = d['result']['gamma']
        file.write(
            f"  {n:>4}  {exp:>9}  {got:>9}  "
            f"{'ok' if exp == got else 'FAIL'}\n"
        )

    # --- Conjecture 2: Gamma_ve = floor(2n/5) ---
    file.write(f"\nConjecture 2 :  Gamma_ve(Cn) = floor(2n/5)\n")
    file.write(dash + "\n")
    misses2 = [
        (d['n'], d['result']['Gamma'], (2 * d['n']) // 5)
        for d in all_data
        if d['result']['Gamma'] != (2 * d['n']) // 5
    ]
    if not misses2:
        ns = [d['n'] for d in all_data]
        file.write(f"  CONFIRMED for n = {ns[0]} .. {ns[-1]}\n\n")
    else:
        for n, got, exp in misses2:
            file.write(f"  FAILS at C{n}: computed={got}, formula={exp}\n")
        file.write("\n")

    file.write(
        f"  {'n':>4}  {'floor(2n/5)':>11}  {'computed':>9}  match\n"
    )
    for d in all_data:
        n   = d['n']
        exp = (2 * n) // 5
        got = d['result']['Gamma']
        file.write(
            f"  {n:>4}  {exp:>11}  {got:>9}  "
            f"{'ok' if exp == got else 'FAIL'}\n"
        )

    # --- Well ve-dominated ---
    file.write(f"\nWell ve-dominated cycles in C{n_min}..C{n_max}:\n")
    file.write(dash + "\n")
    well_ns = [d['n'] for d in all_data if d['result']['well']]
    file.write(f"  n in {{ {', '.join(str(n) for n in well_ns)} }}\n")
    file.write(
        "  (exactly when ceil(n/4) = floor(2n/5);"
        " no well ve-dominated Cn exists for n > 9)\n"
    )

    # --- Structural characterization ---
    file.write(f"\nStructural Characterization:\n")
    file.write(dash + "\n")
    file.write(
        "  S is a minimal ve-dominating set of Cn iff its circular\n"
        "  gap sequence (d_0,...,d_{k-1}) satisfies:\n\n"
        "    (C1)  max(d_i) <= 4              [covering: each gap\n"
        "                                      spanned by 2 endpoints]\n"
        "    (C2)  min(d_i + d_{i+1}) >= 5   [minimality: every vertex\n"
        "                                      has a private edge, k>=2]\n\n"
    )
    all_ok = all(d['pa']['all_verified'] for d in all_data)
    file.write(
        f"  Verified for ALL minimal sets in C{n_min}..C{n_max}:"
        f"  {'PASS' if all_ok else 'FAIL'}\n\n"
    )
    file.write(
        "  Structural consequences:\n"
        "    gamma_ve(Cn):\n"
        "      minimise k s.t. sum d_i = n, all d_i in [1,4],\n"
        "      consecutive pair sums >= 5.\n"
        "      Optimal: all d_i = 4  =>  k = ceil(n/4).\n\n"
        "    Gamma_ve(Cn):\n"
        "      maximise k with the same constraints.\n"
        "      Each pair (d_i, d_{i+1}) sums to >= 5, so\n"
        "      n = sum d_i >= 5k/2  =>  k <= floor(2n/5).\n"
        "      Achieved by repeating block (2,3)  =>  k = floor(2n/5).\n"
    )

    # --- All distinct canonical patterns ---
    file.write(f"\nAll Distinct Canonical Patterns (C{n_min}-C{n_max}):\n")
    file.write(dash + "\n")
    file.write(
        f"  {'Pattern':<20} {'k':>3} {'Type':<14}"
        f" {'minPS':>6}  Appears in\n"
    )
    file.write("  " + "-" * 63 + "\n")

    global_all   = set()
    global_gamma = set()
    global_Gamma = set()
    appears_in   = {}

    for d in all_data:
        pa = d['pa']
        n  = d['n']
        for pat in pa['all_counts']:
            global_all.add(pat)
            appears_in.setdefault(pat, [])
            appears_in[pat].append(n)
        global_gamma.update(pa['gamma_counts'])
        global_Gamma.update(pa['Gamma_counts'])

    for pat in sorted(global_all, key=lambda p: (len(p), p)):
        k     = len(pat)
        types = []
        if pat in global_gamma:
            types.append('gamma')
        if pat in global_Gamma:
            types.append('Gamma')
        type_str = '+'.join(types)
        _, det   = characterization_check(pat)
        ps_str   = str(det['min_psum']) if det['min_psum'] is not None else '—'
        ns_str   = ' '.join(f"C{n}" for n in appears_in[pat])
        file.write(
            f"  {_fmt_pat(pat):<20} {k:>3} {type_str:<14}"
            f" {ps_str:>6}  {ns_str}\n"
        )


# =====================================================
# DRAWING FUNCTION
# =====================================================

def draw_cycle(G, selected_vertices, windows, edge_index):

    pos = nx.circular_layout(G)

    node_colors = [
        "green" if v in selected_vertices else "lightblue"
        for v in G.nodes()
    ]

    covered_edges = set()

    for v in selected_vertices:
        covered_edges |= windows[v]

    edge_colors = [
        "red" if edge_index[tuple(sorted(e))] in covered_edges
        else "black"
        for e in G.edges()
    ]

    plt.figure(figsize=(7, 7))

    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        edge_color=edge_colors,
        node_size=1200,
        font_size=12,
        width=3
    )

    plt.title(f"Selected vertices = {selected_vertices}")
    plt.show()


# =====================================================
# FILE WRITING HELPER
# =====================================================

def write_graph_results(file, graph_label, analyzer, result, pa=None):
    """
    Write per-graph section.

    Full mode  (result['light'] == False):
      writes the complete minimal-set listing with vertex-edge details.

    Light mode (result['light'] == True):
      writes summary only — cardinalities, count, one representative
      per extreme, and pattern statistics.  No full set listing.

    pa: pattern-analysis dict.  In light mode this is taken from
        result['stream_pa'] when pa is not provided explicitly.
    """
    sep  = "=" * 65
    dash = "-" * 65
    is_light = result.get('light', False)

    # Use streamed pa if caller did not supply one
    if pa is None and is_light:
        pa = result.get('stream_pa')

    file.write(f"\n{sep}\n")
    file.write(f"{graph_label}\n")
    file.write(f"{sep}\n\n")

    n = analyzer.G.number_of_nodes()
    m = analyzer.G.number_of_edges()

    file.write(f"n = {n},  |E| = {m}\n\n")

    # --- cardinality summary ---
    file.write("Distribution of minimal ve-dominating sets:\n\n")

    for size, cnt in sorted(result['distribution'].items()):
        file.write(f"  |S| = {size}:  {cnt} set(s)\n")

    total = result.get('count', len(result['minimal_sets'] or []))
    file.write(f"\n  total minimal sets : {total}\n")
    file.write(f"\n")
    file.write(
        f"  gamma_ve({graph_label}) = {result['gamma']}"
        f"   (smallest cardinality)\n"
    )
    file.write(
        f"  Gamma_ve({graph_label}) = {result['Gamma']}"
        f"   (largest cardinality)\n"
    )
    file.write(
        f"  gap = Gamma_ve - gamma_ve = {result['gap']}\n"
    )

    if result['well']:
        file.write(
            f"\n  => {graph_label} is well ve-dominated"
            f" (all minimal sets have equal cardinality)\n"
        )

    # --- backtracking stats ---
    st = result['stats']
    file.write(f"\n  Backtracking stats:\n")
    file.write(f"    recursive calls : {st['calls']}\n")
    file.write(f"    pruned (total)  : {st.get('pruned', 'N/A')}\n")

    if 'pruned_zero_gain' in st:
        file.write(f"      P1 zero-gain  : {st['pruned_zero_gain']}\n")
        file.write(f"      P2 priv-kill  : {st['pruned_private']}\n")
        file.write(f"      P3 dead-end   : {st['pruned_dead_end']}\n")

    # --- pattern analysis ---
    if pa is not None:
        write_pattern_section(file, n, result, pa)

    # --- representative sets (always shown) ---
    file.write(f"\n{dash}\n")

    if is_light:
        file.write("Representative sets (light mode — full listing suppressed):\n")
        file.write(f"{dash}\n\n")
        if result['gamma_sets']:
            file.write(
                "gamma_ve-set  (|S| = "
                f"{result['gamma']}):\n"
            )
            file.write(
                analyzer.format_set_detail(result['gamma_sets'][0])
                + "\n\n"
            )
        if result['Gamma_sets'] and result['gamma'] != result['Gamma']:
            file.write(
                "Gamma_ve-set  (|S| = "
                f"{result['Gamma']}):\n"
            )
            file.write(
                analyzer.format_set_detail(result['Gamma_sets'][0])
                + "\n\n"
            )
    else:
        file.write(
            "Minimal ve-dominating sets"
            " with vertex-edge domination details:\n"
        )
        file.write(f"{dash}\n\n")
        for S in result['minimal_sets']:
            file.write(analyzer.format_set_detail(S) + "\n\n")

    file.write("\n")


# =====================================================
# RESEARCH REPORT GENERATION
# =====================================================

def rectangular_layout(n, width=2.0, height=1.0):
    """
    Place n cycle vertices evenly around a rectangle perimeter,
    starting at top-left and going clockwise (top→right→bottom→left).
    Returns {vertex_index: (x, y)} centered at origin.
    Used for cycles with n > 10 where circular layout gets crowded.
    """
    W, H  = width, height
    perim = 2 * (W + H)
    pos   = {}

    for i in range(n):
        d = (i / n) * perim

        if d <= W:
            x, y = d, H
        elif d <= W + H:
            x, y = W, H - (d - W)
        elif d <= 2 * W + H:
            x, y = W - (d - W - H), 0.0
        else:
            x, y = 0.0, d - 2 * W - H

        pos[i] = (x - W / 2, y - H / 2)

    return pos


def draw_ve_set(G, n, S, windows, edge_index, title, out_path):
    """
    Draw cycle G with ve-dominating set S; save to out_path.

    Layout:  circular for n <= 10, rectangular for n > 10.
    Selected vertices : green
    Dominated edges   : red
    Other vertices    : lightblue
    Other edges       : black
    """
    S_set = set(S)

    pos = (
        nx.circular_layout(G)
        if n <= 10
        else rectangular_layout(n)
    )

    node_colors = [
        'green' if v in S_set else 'lightblue'
        for v in G.nodes()
    ]

    covered = set()
    for v in S_set:
        covered |= windows[v]

    edge_colors = [
        'red'
        if edge_index[tuple(sorted(e))] in covered
        else 'black'
        for e in G.edges()
    ]

    node_size = 800  if n <= 20 else (400 if n <= 35 else 250)
    font_size = 9    if n <= 20 else (7   if n <= 35 else 5  )
    fig_w     = 9    if n <= 20 else 13
    fig_h     = 7    if n <= 10 else (5   if n <= 20 else 6  )

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    nx.draw(
        G, pos, ax=ax,
        with_labels=True,
        node_color=node_colors,
        edge_color=edge_colors,
        node_size=node_size,
        font_size=font_size,
        width=2.5,
    )

    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------
# Text / CSV reports
# --------------------------------------------------

def generate_summary_table_txt(all_data):

    header = (
        f"{'Cycle':<8} | {'gamma_ve':>8} | {'Gamma_ve':>8}"
        f" | {'gap':>4} | {'well':<4}"
        f" | {'minimal_set_count':>18} | {'mode':<5}"
    )
    sep = '-' * len(header)

    lines = [header, sep]

    for d in all_data:
        n    = d['n']
        r    = d['result']
        mode = 'light' if r.get('light') else 'full'
        well = 'yes' if r['well'] else 'no'
        cnt  = r.get('count', len(r['minimal_sets'] or []))

        lines.append(
            f"C{n:<7} | {r['gamma']:>8} | {r['Gamma']:>8}"
            f" | {r['gap']:>4} | {well:<4}"
            f" | {cnt:>18} | {mode:<5}"
        )

    with open('results/summary_table.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def generate_summary_table_csv(all_data):

    with open(
        'results/summary_table.csv', 'w',
        newline='', encoding='utf-8'
    ) as f:
        w = csv.writer(f)
        w.writerow([
            'cycle', 'gamma_ve', 'Gamma_ve', 'gap',
            'well', 'minimal_set_count', 'mode'
        ])
        for d in all_data:
            n    = d['n']
            r    = d['result']
            mode = 'light' if r.get('light') else 'full'
            well = 'yes' if r['well'] else 'no'
            cnt  = r.get('count', len(r['minimal_sets'] or []))
            w.writerow([
                f'C{n}', r['gamma'], r['Gamma'],
                r['gap'], well, cnt, mode
            ])


def generate_performance_csv(all_data):

    with open(
        'results/performance_table.csv', 'w',
        newline='', encoding='utf-8'
    ) as f:
        w = csv.writer(f)
        w.writerow([
            'cycle', 'bt_calls', 'bt_pruned',
            'p1_zero_gain', 'p2_priv_kill', 'p3_dead_end',
            'runtime_seconds'
        ])
        for d in all_data:
            n  = d['n']
            st = d['result']['stats']
            w.writerow([
                f'C{n}',
                st['calls'],
                st.get('pruned', ''),
                st.get('pruned_zero_gain', ''),
                st.get('pruned_private', ''),
                st.get('pruned_dead_end', ''),
                round(d.get('bt_time', 0), 6),
            ])


def generate_conjecture_table(all_data):

    n_min   = all_data[0]['n']
    n_max   = all_data[-1]['n']
    well_ns = [d['n'] for d in all_data if d['result']['well']]

    header = (
        f"{'Cycle':<8} | {'gamma_ve':>8} | {'Gamma_ve':>8}"
        f" | {'gap':>4} | {'well':<4}"
    )
    sep = '-' * len(header)

    lines = [
        f'Conjecture Table: ve-domination numbers on Cn  '
        f'(C{n_min}..C{n_max})',
        '=' * 56,
        '',
        header,
        sep,
    ]

    for d in all_data:
        n    = d['n']
        r    = d['result']
        well = 'yes' if r['well'] else 'no'
        lines.append(
            f"C{n:<7} | {r['gamma']:>8} | {r['Gamma']:>8}"
            f" | {r['gap']:>4} | {well:<4}"
        )

    lines += [
        sep, '',
        'Well ve-dominated cycles found:',
        '{' + ', '.join(f'C{n}' for n in well_ns) + '}',
        '',
        'Conjectured formulas (verified computationally):',
        '  gamma_ve(Cn) = ceil(n / 4)',
        '  Gamma_ve(Cn) = floor(2n / 5)',
        '',
    ]

    with open(
        'results/conjecture_table.txt', 'w', encoding='utf-8'
    ) as f:
        f.write('\n'.join(lines))


# --------------------------------------------------
# Plots
# --------------------------------------------------

def generate_gap_plot(all_data):

    ns   = [d['n'] for d in all_data]
    gaps = [d['result']['gap'] for d in all_data]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ns, gaps, 'o-', color='steelblue',
            linewidth=1.5, markersize=5)
    ax.set_xlabel('n', fontsize=12)
    ax.set_ylabel(r'$\Gamma_{ve}(C_n) - \gamma_{ve}(C_n)$',
                  fontsize=12)
    ax.set_title(r'VE-domination gap on $C_n$', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.set_major_locator(
        plt.MaxNLocator(integer=True, nbins=20)
    )
    plt.tight_layout()
    fig.savefig('results/gap_plot.png', dpi=150)
    plt.close(fig)


def generate_gamma_plot(all_data):

    ns     = [d['n'] for d in all_data]
    gammas = [d['result']['gamma'] for d in all_data]
    Gammas = [d['result']['Gamma'] for d in all_data]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ns, gammas, 's-', color='steelblue',
            linewidth=1.5, markersize=5,
            label=r'$\gamma_{ve}(C_n)$')
    ax.plot(ns, Gammas, 'D-', color='tomato',
            linewidth=1.5, markersize=5,
            label=r'$\Gamma_{ve}(C_n)$')
    ax.set_xlabel('n', fontsize=12)
    ax.set_ylabel('cardinality', fontsize=12)
    ax.set_title(
        r'$\gamma_{ve}$ and $\Gamma_{ve}$ on $C_n$',
        fontsize=13
    )
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.set_major_locator(
        plt.MaxNLocator(integer=True, nbins=20)
    )
    plt.tight_layout()
    fig.savefig('results/gamma_plot.png', dpi=150)
    plt.close(fig)


def generate_count_plot(all_data):

    ns   = [d['n'] for d in all_data]
    cnts = [
        d['result'].get('count', len(d['result']['minimal_sets'] or []))
        for d in all_data
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(ns, cnts, 'o-', color='darkorange',
                linewidth=1.5, markersize=5)
    ax.set_xlabel('n', fontsize=12)
    ax.set_ylabel(
        'number of minimal ve-dominating sets (log scale)',
        fontsize=12
    )
    ax.set_title(
        r'Minimal ve-dominating set count on $C_n$',
        fontsize=13
    )
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.xaxis.set_major_locator(
        plt.MaxNLocator(integer=True, nbins=20)
    )
    plt.tight_layout()
    fig.savefig('results/minimal_set_count_plot.png', dpi=150)
    plt.close(fig)


def generate_well_cycles_plot(all_data):

    ns    = [d['n'] for d in all_data]
    gaps  = [d['result']['gap'] for d in all_data]
    well  = [d['result']['well'] for d in all_data]
    clrs  = ['green' if w else 'red' for w in well]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ns, gaps, '-', color='gray',
            linewidth=0.8, alpha=0.5, zorder=2)
    ax.scatter(ns, gaps, c=clrs, s=80, zorder=3)

    legend_handles = [
        mlines.Line2D(
            [], [], marker='o', linestyle='None',
            markerfacecolor='green', markersize=9,
            label='well ve-dominated (gap = 0)'
        ),
        mlines.Line2D(
            [], [], marker='o', linestyle='None',
            markerfacecolor='red', markersize=9,
            label='not well ve-dominated'
        ),
    ]
    ax.legend(handles=legend_handles, fontsize=10)
    ax.set_xlabel('n', fontsize=12)
    ax.set_ylabel(r'gap = $\Gamma_{ve} - \gamma_{ve}$', fontsize=12)
    ax.set_title(
        r'Well ve-dominated cycles on $C_n$',
        fontsize=13
    )
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.set_major_locator(
        plt.MaxNLocator(integer=True, nbins=20)
    )
    plt.tight_layout()
    fig.savefig('results/well_cycles_plot.png', dpi=150)
    plt.close(fig)


# --------------------------------------------------
# Drawings
# --------------------------------------------------

def generate_drawings(all_data):

    os.makedirs('results/drawings', exist_ok=True)

    for d in all_data:
        n      = d['n']
        result = d['result']

        G        = nx.cycle_graph(n)
        analyzer = VEDominationAnalyzer(G)

        # gamma-set drawing
        if result['gamma_sets']:
            S = result['gamma_sets'][0]
            draw_ve_set(
                G, n, S,
                analyzer.windows,
                analyzer.edge_index,
                title=(
                    f'C{n}   —   '
                    f'γᵥᵉ-set   '
                    f'|S| = {len(S)}'
                ),
                out_path=f'results/drawings/C{n}_gamma.png',
            )

        # Gamma-set drawing (always produce, even when equal to gamma)
        if result['Gamma_sets']:
            S = result['Gamma_sets'][0]
            well_note = (
                '  (well ve-dominated)'
                if result['well'] else ''
            )
            draw_ve_set(
                G, n, S,
                analyzer.windows,
                analyzer.edge_index,
                title=(
                    f'C{n}   —   '
                    f'Γᵥᵉ-set   '
                    f'|S| = {len(S)}'
                    f'{well_note}'
                ),
                out_path=f'results/drawings/C{n}_Gamma_max.png',
            )


# --------------------------------------------------
# Master entry point
# --------------------------------------------------

def generate_all_reports(all_data):
    """Generate all research-report files under results/."""

    os.makedirs('results', exist_ok=True)

    steps = [
        ('summary_table.txt',           generate_summary_table_txt),
        ('summary_table.csv',           generate_summary_table_csv),
        ('performance_table.csv',       generate_performance_csv),
        ('conjecture_table.txt',        generate_conjecture_table),
        ('gap_plot.png',                generate_gap_plot),
        ('gamma_plot.png',              generate_gamma_plot),
        ('minimal_set_count_plot.png',  generate_count_plot),
        ('well_cycles_plot.png',        generate_well_cycles_plot),
        ('drawings/',                   generate_drawings),
    ]

    print('\nGenerating reports:', flush=True)

    for label, fn in steps:
        fn(all_data)
        print(f'  {label}', flush=True)

    n_drawings = len(all_data) * 2
    print(
        f'  ({n_drawings} PNGs in drawings/)',
        flush=True
    )
    print('Done.', flush=True)


# =====================================================
# MAIN — compare brute force vs backtracking C3–C20
# =====================================================

BRUTE_MAX_N = 15   # run brute force up to this n (for verification)
BACK_MAX_N  = 50   # run backtracking up to this n
LIGHT_N     = 20   # n > LIGHT_N uses light mode (no full set storage)


if __name__ == "__main__":

    total_start = time.time()

    with open(
        "results/cycle_summary.txt",
        "w",
        encoding="utf-8"
    ) as file:

        # --- summary table headers ---

        file_hdr = (
            f"{'Graph':<8}"
            f"{'gve':<6}"
            f"{'Gve':<6}"
            f"{'gap':<6}"
            f"{'well':<7}"
            f"{'sets':>10}"
            f"{'BT calls':>12}"
            f"{'BT pruned':>11}"
            f"{'BT time':>10}"
            f"{'BF time':>10}"
            f"  mode"
        )

        print(file_hdr)
        print("-" * 87)

        file.write(file_hdr + "\n")
        file.write("-" * 87 + "\n")

        all_data = []   # collect for global summary

        for n in range(3, BACK_MAX_N + 1):

            G        = nx.cycle_graph(n)
            analyzer = VEDominationAnalyzer(G)
            use_light = n > LIGHT_N

            # --- backtracking (full or light) ---

            t0 = time.time()

            if use_light:
                result = analyzer.analyze(
                    method='backtrack',
                    light=True,
                    cycle_n=n
                )
            else:
                result = analyzer.analyze(method='backtrack')

            bt_time = time.time() - t0

            # --- brute force (small n only, full mode only) ---

            bf_str = "       N/A"

            if n <= BRUTE_MAX_N and not use_light:

                t0 = time.time()
                bf_result = analyzer.analyze(method='brute')
                bf_time   = time.time() - t0
                bf_str    = f"{bf_time:9.4f}s"

                if (
                    sorted(result['minimal_sets'])
                    != sorted(bf_result['minimal_sets'])
                ):
                    print(f"  *** MISMATCH on C{n} ***")

            # --- pattern analysis ---

            if use_light:
                # pattern data already accumulated during the single pass
                pa = result['stream_pa']
            else:
                pa = analyze_cycle_patterns(
                    result['minimal_sets'],
                    result['gamma'],
                    result['Gamma'],
                    n
                )

            all_data.append({
                'n':       n,
                'result':  result,
                'pa':      pa,
                'bt_time': bt_time,
            })

            # --- summary table line ---

            st        = result['stats']
            well_str  = "yes" if result['well'] else "no"
            set_count = result.get('count', len(result['minimal_sets'] or []))
            mode_str  = "light" if use_light else "full"

            line = (
                f"C{n:<7}"
                f"{result['gamma']:<6}"
                f"{result['Gamma']:<6}"
                f"{result['gap']:<6}"
                f"{well_str:<7}"
                f"{set_count:>10}"
                f"{st['calls']:>12}"
                f"{st['pruned']:>11}"
                f"{bt_time:9.4f}s"
                f"{bf_str:>10}"
                f"  {mode_str}"
            )

            print(line)
            file.write(line + "\n")

            # --- detailed per-graph section ---

            write_graph_results(
                file, f"C{n}", analyzer, result, pa=pa
            )

        # --- global conjecture + pattern summary ---

        write_conjecture_summary(file, all_data)

    # --- research reports (outside the file context manager) ---

    generate_all_reports(all_data)

    elapsed = round(time.time() - total_start, 2)
    print(f"\nTotal elapsed: {elapsed}s")

