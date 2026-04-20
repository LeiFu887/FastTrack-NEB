import csv
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atom
from FastTrack.config import Calculator
try:
    from FastTrack.config import predict_batch
except ImportError:
    predict_batch = None
import numpy as np

def compute(ion_pos, nearest_ion_pos, input_file, atom_type, output, batch_size=128):
    
    struc = Structure.from_file(input_file)  
    adp = AseAtomsAdaptor()
    atoms = adp.get_atoms(struc)

    xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz ,nearest_ion_pos_wrapped= _get_wrapping_box(atoms, ion_pos, nearest_ion_pos)

    calc = Calculator()
    atoms.calc = calc
    results = []

    grid_points = []
    valid_mask = []
    for k in range(nz):
        for i in range(nx):
            for j in range(ny):
                z = zmin + (k * (zmax - zmin) / nz)
                x = xmin + (i * (xmax - xmin) / nx)
                y = ymin + (j * (ymax - ymin) / ny)
                cart_pos = np.array([x, y, z], dtype=float)
                distances = np.linalg.norm(atoms.positions - cart_pos, axis=1)
                grid_points.append(cart_pos)
                valid_mask.append(bool(np.min(distances) >= 0.5))

    if predict_batch is not None:
        host = adp.get_structure(atoms)
        valid_structures = []
        valid_indices = []
        for idx, is_valid in enumerate(valid_mask):
            if not is_valid:
                continue
            s = host.copy()
            s.append(atom_type, grid_points[idx], coords_are_cartesian=True)
            valid_structures.append(adp.get_atoms(s))
            valid_indices.append(idx)

        pred_by_idx = {}
        if valid_structures:
            energies = predict_batch(calc, valid_structures, batch_size=batch_size)
            for local_i, global_i in enumerate(valid_indices):
                pred_by_idx[global_i] = float(energies[local_i])

        for idx, cart_pos in enumerate(grid_points):
            x, y, z = cart_pos.tolist()
            if not valid_mask[idx]:
                results.append([x, y, z, 0.0])
                continue
            e = pred_by_idx.get(idx, 0.0)
            results.append([x, y, z, e if e < 0 else 0.0])
    else:
        for idx, cart_pos in enumerate(grid_points):
            x, y, z = cart_pos.tolist()
            if not valid_mask[idx]:
                results.append([x, y, z, 0.0])
                continue
            atoms.append(Atom(atom_type, position=cart_pos))
            e = atoms.get_potential_energy()
            atoms.pop()
            results.append([x, y, z, e if e < 0 else 0.0])

    with open(output, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)   
        writer.writerow(["x", "y", "z", "energy"])
        writer.writerows(results)
    
    return nearest_ion_pos_wrapped

def _get_wrapping_box(atoms, ion_pos, nearest_ion_pos, padding=1.3):
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    ion_frac = np.linalg.solve(cell.T, ion_pos)
    nearest_ion_frac = np.linalg.solve(cell.T, nearest_ion_pos)

    
    for i in range(3):
        if pbc[i]:
            delta = nearest_ion_frac[i] - ion_frac[i]
            if delta > 0.5:
                nearest_ion_frac[i] -= 1.0
            elif delta < -0.5:
                nearest_ion_frac[i] += 1.0

    
    nearest_ion_pos_wrapped = np.dot(nearest_ion_frac, cell)
    
    xmin = min(ion_pos[0], nearest_ion_pos_wrapped[0]) - padding
    xmax = max(ion_pos[0], nearest_ion_pos_wrapped[0]) + padding
    ymin = min(ion_pos[1], nearest_ion_pos_wrapped[1]) - padding
    ymax = max(ion_pos[1], nearest_ion_pos_wrapped[1]) + padding
    zmin = min(ion_pos[2], nearest_ion_pos_wrapped[2]) - padding
    zmax = max(ion_pos[2], nearest_ion_pos_wrapped[2]) + padding
    nx = 12+ int(abs(ion_pos[0]-nearest_ion_pos_wrapped[0]) * 5)
    ny = 12+ int(abs(ion_pos[1]-nearest_ion_pos_wrapped[1]) * 5)
    nz = 12+ int(abs(ion_pos[2]-nearest_ion_pos_wrapped[2]) * 5)

    return xmin, xmax, ymin, ymax, zmin, zmax , nx , ny ,nz ,nearest_ion_pos_wrapped
