import os

from FastTrack.structure_processing import *
from FastTrack.Calculator import *
from FastTrack.visualization import *
from FastTrack.neb_c import *
from FastTrack.Ionremove import *
from FastTrack.full_connected import *


# ---------------------------------------------------------------------------
# Lightweight formatting helpers for console output.
# ---------------------------------------------------------------------------
_LINE_W = 72


def _banner(title: str, char: str = "="):
    print(char * _LINE_W, flush=True)
    print(title, flush=True)
    print(char * _LINE_W, flush=True)


def _rule(char: str = "-"):
    print(char * _LINE_W, flush=True)


def _fmt_vec(v):
    return "[" + ", ".join(f"{float(x):+.3f}" for x in v) + "]"


# ---------------------------------------------------------------------------
# Top-level pipelines
# ---------------------------------------------------------------------------

def min_barrier(input_file, atom_type, atom_env):
    """
    Minimum single-ion migration barrier: screens all distinct hops
    emanating from a representative migrating ion and returns the lowest
    relaxed barrier.

    Args:
        input_file(str): Structure file (cif / POSCAR / ...).
        atom_type(str):  Migrating ion species, e.g. 'Li'.
        atom_env(int):   1 - remove 2 ions (target + nearest).
                         2 - remove 3 ions (target + two nearest).
                         0 - keep only target and nearest; all other
                             same-species ions removed (dilute limit).

    Example:
        min_barrier("LiCoO2.cif", 'Li', 1)
    """
    _banner("FastTrack :: min_barrier  --  single-ion minimum migration barrier")

    scale_factors = expand_cell_by_lattice_length(input_file, "POSCAR-expand")
    print(f"[1/4] Supercell expansion     : scale factor (a,b,c) = {tuple(scale_factors)}", flush=True)

    optimizer = StructureOptimizer("POSCAR-expand", "POSCAR-op", fmax=0.01, steps=300)
    optimizer.optimize_structure()
    print("[2/4] Structure optimization  : -> POSCAR-op", flush=True)

    env_paths = env_path_finder("POSCAR-op", atom_type)
    print(f"[3/4] Distinct migration hops : {len(env_paths)} path(s) found", flush=True)

    print("[4/4] Per-hop barrier evaluation", flush=True)
    max_e_op_s = []
    for i, env_path in enumerate(env_paths):
        dir_name = f"path{i + 1}"
        os.makedirs(dir_name, exist_ok=True)
        _, max_e_op = hw("POSCAR-op", atom_type, atom_env, dir_name, env_path)
        max_e_op_s.append(max_e_op)

    eb = min(max_e_op_s)
    _banner(f"FastTrack :: min_barrier done  --  Minimum relaxed barrier = {eb:.4f} eV")
    return eb


def hw(input_file, atom_type, atom_env, dir_name, env_path,
       batch_size=128, pes_visualization=True):
    """
    Complete per-hop pipeline:
        ion removal -> endpoint relaxation -> PES sampling ->
        fixed-lattice path optimization -> relaxed-path evaluation.
    Returns (max_e, max_e_op).
    """
    _rule()
    print(f"[hw] dir={dir_name}", flush=True)

    output_file_rm = os.path.join(dir_name, "POSCAR_rm")

    remover = IonRemover(input_file, atom_type, env_path)
    if atom_env == 1:
        remover.if_2()
        ion_pos, nearest_ion_pos = remover.remove_two_ions(output_file_rm)
    elif atom_env == 2:
        ion_pos, nearest_ion_pos = remover.remove_three_ions(output_file_rm)
    elif atom_env == 0:
        output_file_0 = os.path.join(dir_name, "POSCAR_keep_two")
        ion_pos, nearest_ion_pos = remover.keep_two_closest_ions(output_file=output_file_0)
        StructureOptimizer(output_file_0, output_file_rm, fmax=0.01, steps=300).optimize_structure()
    else:
        raise ValueError("atom_env must be 0, 1 or 2")
    print(f"  - Ion removal            : -> {output_file_rm}", flush=True)

    output_file_1 = os.path.join(dir_name, "POSCAR")
    optimizer = StructureOptimizer(output_file_rm, output_file_1, fmax=0.01, steps=300)
    e1, ion_pos_op = optimizer.optimize_with_fix_and_add_atom(atom_type, ion_pos)
    optimizer2 = StructureOptimizer(output_file_rm, output_file_1, fmax=0.01, steps=300)
    e2, nearest_ion_pos_op = optimizer2.optimize_with_fix_and_add_atom(atom_type, nearest_ion_pos)
    min_energy = min(e1, e2)
    print(f"  - Endpoint relaxation    : E_init = {e1:.4f} eV, E_final = {e2:.4f} eV", flush=True)
    print(f"  - Optimized initial pos  : {_fmt_vec(ion_pos_op)}", flush=True)
    print(f"  - Optimized final   pos  : {_fmt_vec(nearest_ion_pos_op)}", flush=True)

    output_potential = os.path.join(dir_name, "potential.csv")
    nearest_ion_pos_wrapped = compute(
        ion_pos_op, nearest_ion_pos_op,
        input_file=output_file_rm,
        atom_type=atom_type,
        output=output_potential,
        batch_size=batch_size,
    )
    print(f"  - PES sampling           : -> {output_potential}", flush=True)
    print(f"  - PBC-wrapped final pos  : {_fmt_vec(nearest_ion_pos_wrapped)}", flush=True)

    output_path = os.path.join(dir_name, "optimized_path.csv")
    max_e, path, e0 = Migration_Energy(
        output_potential, ion_pos_op, nearest_ion_pos_wrapped,
        num_points=31, iteration_=30,
        output=output_path,
    )

    out_pic_op = os.path.join(dir_name, "Migration_Energy_op.jpg")
    out_xsf = os.path.join(dir_name, "Migration.xsf")
    max_e_op = Migration_Energy_op(
        path, atom_type, 2.8, out_pic_op, output_file_rm, out_xsf, e0,
    )

    print(f"  - Migration barriers     : fixed = {max_e:.4f} eV, relaxed = {max_e_op:.4f} eV", flush=True)

    if pes_visualization:
        out_html = os.path.join(dir_name, "Energy_Isosurfaces.html")
        potential_draw(max_e + 0.1, min_energy, output_potential, output_path, out_html)
        print(f"  - PES visualization      : -> {out_html}", flush=True)

    return max_e, max_e_op


def optimize_struc(input_file, output_file='POSCAR-out', fmax=0.03, steps=300):
    optimizer = StructureOptimizer(input_file=input_file, output_file=output_file, fmax=fmax, steps=steps)
    optimizer.optimize_structure()


def neb(initial_struc, final_struc, nimages=7, fmax=0.03, steps=300):
    """NEB with MLFF."""
    _banner("FastTrack :: neb")
    print("[1/3] Endpoint optimization", flush=True)
    StructureOptimizer(initial_struc, "POSCAR_initial", fmax, steps).optimize_fix()
    StructureOptimizer(final_struc, "POSCAR_final", fmax, steps).optimize_fix()

    print("[2/3] NEB run", flush=True)
    success = nebrun_c(nimages, fmax)

    if success:
        print("[3/3] NEB converged", flush=True)
        traj_plt_c(nimages)
    else:
        print("[3/3] NEB did NOT converge", flush=True)


def perc_barrier(input_file, atom_type, atom_env, batch_size=128, pes_visualization=False):
    """
    Global percolation bottleneck barrier.

    Finds the minimum value ``thr`` such that, keeping only hops with
    relaxed barrier <= ``thr``, the migrating species can still form a
    connected pathway that crosses a periodic image (i.e. percolates
    through the lattice).
    """
    _banner("FastTrack :: perc_barrier  --  global percolation bottleneck barrier")

    scale_factors = expand_cell_by_lattice_length(input_file, "POSCAR-expand")
    print(f"[1/5] Supercell expansion     : scale factor (a,b,c) = {tuple(scale_factors)}", flush=True)

    optimizer = StructureOptimizer("POSCAR-expand", "POSCAR-op", fmax=0.01, steps=300)
    optimizer.optimize_structure()
    print("[2/5] Structure optimization  : -> POSCAR-op", flush=True)

    structure_file = "POSCAR-op"
    migrating_specie = atom_type

    print("[3/4] Unique-hop barriers     : building migration graph & evaluating ...", flush=True)
    mg, barrier_by_label = compute_all_hop_barriers(
        structure_file=structure_file,
        atom_type=migrating_specie,
        atom_env=atom_env,
        hop_barrier_fn=hw,
        expand_factor=1.2,
        batch_size=batch_size,
        pes_visualization=pes_visualization,
    )

    print("[4/4] Percolation search      : threshold scanning ...", flush=True)
    thr, (u, hop_list) = find_min_percolating_path(mg, barrier_by_label)

    _rule()
    print(f"Percolation bottleneck threshold = {thr:.4f} eV", flush=True)
    print_path(mg, u, hop_list)
    export_path_to_cif(structure_file, hop_list, migrating_specie)

    _banner(f"FastTrack :: perc_barrier done  --  Bottleneck barrier = {thr:.4f} eV")
    return thr
