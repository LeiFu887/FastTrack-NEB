import itertools
import math
import heapq
import os
from copy import deepcopy

import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.diffusion.neb.full_path_mapper import MigrationGraph, order_path
from pymatgen.analysis.diffusion.neb.pathfinder import DistinctPathFinder, MigrationHop
from pymatgen.analysis.diffusion.neb.periodic_dijkstra import _get_adjacency_with_images


def _get_site_idx(all_sites, spec_site, tol=1e-8):
    idx = None
    for i, site in enumerate(all_sites):
        if site.specie == spec_site.specie and spec_site.distance(site) < tol:
            idx = i
            break
    return idx


def compute_all_hop_barriers(
    structure_file: str,
    atom_type: str,
    atom_env: int,
    hop_barrier_fn,
    expand_factor: float = 1.2,
    batch_size: int = 128,
    pes_visualization: bool = False,
):
    """
    Build ONE MigrationGraph with
        max_distance = DistinctPathFinder.max_path_length * expand_factor
    and compute the migration barrier for every unique hop exactly once.

    Rationale:
        `MigrationGraph.hop_label` is assigned in the order that equivalence
        classes are first seen while iterating the graph's edges (see
        `_group_and_label_hops` in pymatgen-analysis-diffusion). That ordering
        is NOT stable across two `with_distance(max_distance=D)` calls with
        different `D`, because MinimumDistanceNN inserts new edges in the
        middle of each site's neighbour list. Therefore it is NOT safe to
        compute barriers on a small-distance graph and then reuse those label
        keys on a larger-distance graph. Building a single graph with the
        final expanded cutoff and computing every hop in it avoids the
        problem entirely.

    hop_barrier_fn: callable with the same signature as FastTrack.hw, used as
        a dependency-injected "one-hop barrier" evaluator to keep this module
        free of circular imports.

    Returns:
        (mg, barrier_by_label)
            mg: MigrationGraph with 'barrier' and 'cost' attached to every edge.
            barrier_by_label: {hop_label: relaxed_barrier_eV}
    """
    structure = Structure.from_file(structure_file)
    try:
        structure.remove_oxidation_states()
    except Exception:
        pass

    path_finder = DistinctPathFinder(structure, migrating_specie=atom_type)
    max_distance = float(path_finder.max_path_length) * float(expand_factor)
    mg = MigrationGraph.with_distance(
        structure, migrating_specie=atom_type, max_distance=max_distance
    )

    print(f"  DistinctPathFinder max_path_length = {path_finder.max_path_length:.3f} A", flush=True)
    print(f"  MigrationGraph max_distance        = {max_distance:.3f} A "
          f"(= {path_finder.max_path_length:.3f} x {expand_factor:g})", flush=True)
    print(f"  Unique hops to evaluate            = {len(mg.unique_hops)}", flush=True)

    sites = structure.sites
    barrier_by_label: dict[int, float] = {}
    for lab, info in mg.unique_hops.items():
        rep_hop: MigrationHop = info["hop"]

        isite_idx = _get_site_idx(sites, rep_hop.isite, tol=1e-5)
        esite_idx = _get_site_idx(sites, rep_hop.esite, tol=1e-5)
        if isite_idx is None or esite_idx is None:
            raise RuntimeError(f"Cannot map endpoints for hop_label={lab}.")

        dir_name = f"hop_{lab}"
        os.makedirs(dir_name, exist_ok=True)

        _, max_e_op = hop_barrier_fn(
            input_file=structure_file,
            atom_type=atom_type,
            atom_env=atom_env,
            dir_name=dir_name,
            env_path=[isite_idx, esite_idx],
            batch_size=batch_size,
            pes_visualization=pes_visualization,
        )
        barrier = float(max_e_op)
        barrier_by_label[lab] = barrier
        mg.add_data_to_similar_edges(lab, {"barrier": barrier})
        print(
            f"  hop_label={lab:<3d}  path={isite_idx}->{esite_idx}"
            f"   barrier={barrier:.4f} eV"
            f"   length={rep_hop.length:.3f} A",
            flush=True,
        )

    mg.assign_cost_to_graph(cost_keys=["barrier"])
    return mg, barrier_by_label


def get_path_bottleneck(mg, max_val, max_image=6, flip_hops=True):
    G0 = deepcopy(mg.m_graph.graph)
    cut_edges = [(u, v, k) for u, v, k, d in G0.edges(keys=True, data=True) if d.get("cost", math.inf) > max_val]
    for u, v, k in cut_edges:
        G0.remove_edge(u, v, key=k)

    conn_dict = _get_adjacency_with_images(G0.to_undirected())

    for start_u in G0.nodes():
        best = {}
        parent = {}
        pq = []
        src = (start_u, (0, 0, 0))
        heapq.heappush(pq, (0.0, src))
        best[src] = 0.0

        while pq:
            cur_cost, (cur_idx, cur_img) = heapq.heappop(pq)
            if cur_cost != best.get((cur_idx, cur_img), math.inf):
                continue

            for nxt_idx, keyed_data in conn_dict[cur_idx].items():
                for d in keyed_data.values():
                    edge_cost = float(d["cost"])
                    new_img = tuple(np.add(cur_img, d["to_jimage"]))
                    if any(abs(x) > max_image for x in new_img):
                        continue
                    nxt = (nxt_idx, new_img)
                    new_cost = max(cur_cost, edge_cost)
                    if new_cost < best.get(nxt, math.inf):
                        best[nxt] = new_cost
                        parent[nxt] = (cur_idx, cur_img)
                        heapq.heappush(pq, (new_cost, nxt))

        targets = [(start_u, img) for (idx, img) in best.keys() if idx == start_u and img != (0, 0, 0)]
        if not targets:
            continue
        leaf = min(targets, key=lambda t: best[t])

        path = []
        cur = leaf
        while cur in parent:
            path.append(cur)
            cur = parent[cur]
        path.append(cur)
        path = path[::-1]

        path_hops = []
        for (idx1, jimg1), (idx2, jimg2) in itertools.pairwise(path):
            i1_, i2_ = sorted((idx1, idx2))
            all_edge_data = list(G0.get_edge_data(i1_, i2_, default={}).items())
            image_diff = np.subtract(jimg2, jimg1)
            found = 0
            for _k, tmp_d in all_edge_data:
                if tmp_d["to_jimage"] in {tuple(image_diff), tuple(-image_diff)}:
                    path_hops.append(tmp_d)
                    found += 1
            if found != 1:
                raise RuntimeError("More than one edge matched in original graph.")

        if flip_hops:
            yield start_u, order_path(path_hops, start_u)
        else:
            yield start_u, path_hops


def find_min_percolating_path(mg, barrier_by_label):
    vals = sorted({float(v) for v in barrier_by_label.values() if v is not None and not np.isnan(v)})
    if not vals:
        raise RuntimeError("barrier_by_label is empty or NaN-only.")

    best_threshold = None
    best_path = None
    for thr in vals:
        paths = list(get_path_bottleneck(mg, max_val=thr, max_image=6, flip_hops=True))
        if not paths:
            continue
        best_threshold = thr
        best_path = min(paths, key=lambda x: max(h["cost"] for h in x[1]))
        break

    if best_threshold is None:
        raise RuntimeError("No percolating pathway found within all thresholds.")
    return best_threshold, best_path


def print_path(mg, start_node, hop_list):
    print(f"start node u = {start_node}")
    print("percolating pathway hops:")
    for i, h in enumerate(hop_list, 1):
        hop = h["hop"]
        lab = h["hop_label"]
        barrier = h.get("barrier", None)
        i_frac = hop.isite.frac_coords
        e_frac = hop.esite.frac_coords
        print(
            f"{i:02d}. label={lab} "
            f"[{i_frac[0]:.3f},{i_frac[1]:.3f},{i_frac[2]:.3f}] -> "
            f"[{e_frac[0]:.3f},{e_frac[1]:.3f},{e_frac[2]:.3f}] "
            f"barrier={barrier:.3f} eV to_jimage={h.get('to_jimage')}"
        )
