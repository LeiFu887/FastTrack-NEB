# FastTrack

## Introduction

FastTrack is a fast, accurate, and flexible framework for predicting atomic
diffusion barriers in crystalline solids. It combines a universal
machine-learning force field (MLFF) with three-dimensional potential-energy-
surface (PES) sampling and interpolation, giving you:

- **`min_barrier`**: the minimum migration barrier of a single ion,
  screened across all distinct hops emanating from it.
- **`perc_barrier`**: the global *percolation bottleneck* barrier — the
  smallest threshold such that the migrating species still forms a
  connected pathway across periodic images.
- **`neb`**: a convenience NEB wrapper on top of the same MLFF.
- **`optimize_struc`**: a convenience cell + atom relaxer.

FastTrack is MLFF-agnostic: GPTFF, UPET, CHGNet, MACE, MatterSim, SevenNet,
or anything else exposing an ASE calculator interface will work.

---

## Installation

```bash
git clone https://github.com/atomly-materials-research-lab/FastTrack.git
cd FastTrack
# Configure your force field first (see next section), then:
pip install .
```

Development install:

```bash
pip install -e .
```

---

## Step 1 — Configure your MLFF in `FastTrack/config.py`

**This step is mandatory.** FastTrack never hard-codes a backend; you decide
which MLFF to use by editing `FastTrack/config.py`:

```python
# FastTrack/config.py
from gptff.model.mpredict import ASECalculator
#from upet.calculator import UPETCalculator
#from chgnet.model import CHGNetCalculator
#from mace.calculators import mace_mp
#from mattersim.forcefield import MatterSimCalculator
#from sevenn.calculator import SevenNetCalculator

def Calculator():
    # --- default: GPTFF ---
    MODEL_WEIGHT = "yourpath/gptff_v1.pth"
    DEVICE = "cuda"
    calc = ASECalculator(MODEL_WEIGHT, DEVICE)

    # --- UPET ---
    # calc = UPETCalculator(model="pet-mad-s", version="1.0.2", device="cuda")

    # --- CHGNet ---
    # calc = CHGNetCalculator()

    # --- MACE ---
    # calc = mace_mp(model="yourpath", default_dtype="float64", device="cuda")

    # --- MatterSim ---
    # calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device="cuda")

    # --- SevenNet ---
    # calc = SevenNetCalculator('7net-mf-ompa', modal='mpa')

    return calc
```

`Calculator()` must return **any** ASE-compatible calculator object. FastTrack
will use `atoms.get_potential_energy()` on it wherever needed.

### Optional — batched PES inference

The PES scan in `compute()` evaluates thousands of grid points per hop. If
your MLFF supports batched energy inference, FastTrack can run it in large
chunks instead of one-by-one, giving an order-of-magnitude speedup.

To opt in, define `predict_batch(calc, atoms_list, batch_size)` at the bottom
of `config.py`. Its job is to translate FastTrack's request (a list of
`ase.Atoms`) into your backend's native batch API and return a list of
energies:

```python
def predict_batch(calc, atoms_list, batch_size=128):
    # --- GPTFF (default) ---
    return calc.predict_energies_batched(
        atoms_list, batch_size=batch_size, show_progress=False
    )

    # --- UPET ---
    # energies = []
    # for i in range(0, len(atoms_list), batch_size):
    #     res = calc.compute_energy(
    #         atoms_list[i:i + batch_size],
    #         compute_forces_and_stresses=False,
    #     )
    #     energies.extend(res["energy"])
    # return energies
```

If you **delete** `predict_batch` (or your MLFF just doesn't have a batch
API), FastTrack transparently falls back to the point-by-point ASE loop.
Everything still works, just slower.

---

## Step 2 — Run FastTrack

After `pip install`, the package exposes four top-level entry points.

### 2.1 `min_barrier` — Single-ion minimum migration barrier

Pick one migrating ion, compute the barrier of every distinct hop around it,
and return the minimum relaxed barrier.

```python
from FastTrack import min_barrier

# atom_env: 1 = two-vacancy (target + nearest), typical lithiated regime
# atom_env: 2 = three-vacancy (for less-ideal midpoints, auto-suggested
#               when hw() detects asymmetry of the mid-triangle)
# atom_env: 0 = dilute limit (only target + nearest kept, all other
#               same-species ions removed)
eb = min_barrier("LiFePO4.cif", 'Li', 1)    # lithiation limit
# or
eb = min_barrier("LiFePO4.cif", 'Li', 0)    # delithiation limit
```

**What `min_barrier` produces (per-run, at the CWD):**

```
POSCAR-expand                 # supercell with all |a|,|b|,|c| >= 10 A
POSCAR-op                     # MLFF-optimized supercell
path1/
  POSCAR_rm                   # structure after ion removal
  POSCAR                      # structure with the migrating atom re-inserted & relaxed
  potential.csv               # 3D PES samples (x, y, z, energy)
  optimized_path.csv          # fixed-lattice minimum-energy path
  Migration_Energy_op.jpg     # energy profile (fixed + relaxed, smoothed)
  Migration.xsf               # trajectory of the relaxed path (XCrysDen)
  Energy_Isosurfaces.html     # interactive 3D iso-surfaces (if pes_visualization)
path2/
  ...                          # one directory per distinct hop
```

### 2.2 `perc_barrier` — Global percolation bottleneck

Compute the minimum barrier threshold at which the migrating species still
percolates across periodic images. Useful for screening bulk ionic
conductivity.

```python
from FastTrack import perc_barrier

thr = perc_barrier("Li6PS5Cl.cif", 'Li', atom_env=1)
```

**What `perc_barrier` produces:**

```
POSCAR-expand, POSCAR-op      # same as min_barrier
hop_<label>/                  # one directory per unique migration hop
                              # (labels come from MigrationGraph.unique_hops)
percolation_path.cif          # the bottleneck pathway drawn on the host lattice
                              # (migrating species at hop endpoints, H as midpoints)
```

The function returns the bottleneck barrier `thr` (eV), and prints the full
ordered hop list with labels, fractional coordinates, per-hop barriers and
periodic images.

### 2.3 `neb` — MLFF-driven NEB

```python
from FastTrack import neb

neb("POSCAR_initial.vasp", "POSCAR_final.vasp", nimages=7, fmax=0.03)
```

Produces `POSCAR_initial`, `POSCAR_final`, `A2B.traj`, `Trajectory.xsf`,
`MEB.png`.

### 2.4 `optimize_struc` — Cell + atom relaxation

```python
from FastTrack import optimize_struc

optimize_struc("LiCoO2.cif", output_file="POSCAR-out", fmax=0.03)
```

---

## Citation

```bibtex
@article{Kang2025FastTrack,
  title     = {FastTrack: A fast method to evaluate mass transport in solid leveraging universal machine learning interatomic potential},
  author    = {Kang, Hanwen and Lu, Tenglong and Qi, Zhanbin and Guo, Jiandong and Meng, Sheng and Liu, Miao},
  journal   = {AI for Science},
  volume    = {1},
  pages     = {015004},
  year      = {2025},
  publisher = {IOP Publishing},
  doi       = {10.1088/3050-287X/ae0808},
  url       = {https://doi.org/10.1088/3050-287X/ae0808}
}
```
