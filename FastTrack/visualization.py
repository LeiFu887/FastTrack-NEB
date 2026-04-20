import numpy as np
import pandas as pd
from ase.io import read, write
import os
from ase.constraints import FixAtoms
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator, make_interp_spline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from FastTrack.structure_processing import optimize_structure_with_fixed_atom
from pymatgen.core import Structure


def optimize_point_in_grid(idx, path,interpolator, start_point, end_point, regularization_coeff=0.1):
    grid_size = 0.6
    grid_points = 50
    
    current_point = path[idx]
    line_direction = (end_point - start_point) / np.linalg.norm(end_point - start_point)  
    plane_normal = line_direction  
    
    u = np.cross(plane_normal, np.array([1, 0, 0]))  
    if np.linalg.norm(u) < 1e-8:
        u = np.cross(plane_normal, np.array([0, 1, 0]))  
    u /= np.linalg.norm(u)
    v = np.cross(plane_normal, u)
    
   
    grid_range = np.linspace(-grid_size / 2, grid_size / 2, grid_points)
    grid_u, grid_v = np.meshgrid(grid_range, grid_range)
    
    
    grid_points_3d = (
        current_point
        + grid_u[..., None] * u
        + grid_v[..., None] * v
    )
    grid_points_3d = grid_points_3d.reshape(-1, 3)  
    
   
    grid_energies = interpolator(grid_points_3d)
    
    
    prev_point = path[idx - 1] if idx > 0 else current_point 
    next_point = path[idx + 1] if idx < len(path) - 1 else current_point 
    
    regularization_term = regularization_coeff * (
        np.linalg.norm(grid_points_3d - prev_point, axis=1) +
        np.linalg.norm(grid_points_3d - next_point, axis=1)
    )
    
    
    total_energies = grid_energies + regularization_term
    
    
    min_energy_idx = np.argmin(total_energies)
    min_energy_point = grid_points_3d[min_energy_idx]
    
    return min_energy_point


def optimize_point_with_neighbors(idx, path, interpolator, grid_size=0.4, grid_points=50, regularization_coeff=0.1):
    """
    
    """
    prev_point = path[idx - 1]
    next_point = path[idx + 1]
    
    
    midpoint = (prev_point + next_point) / 2
    
    
    line_direction = next_point - prev_point
    line_direction /= np.linalg.norm(line_direction)  
    
    
    u = np.cross(line_direction, np.array([1, 0, 0])) 
    if np.linalg.norm(u) < 1e-8:
        u = np.cross(line_direction, np.array([0, 1, 0]))  
    u /= np.linalg.norm(u)
    v = np.cross(line_direction, u)  
    
    
    grid_range = np.linspace(-grid_size / 2, grid_size / 2, grid_points)
    grid_u, grid_v = np.meshgrid(grid_range, grid_range)
    
   
    grid_points_3d = (
        midpoint
        + grid_u[..., None] * u
        + grid_v[..., None] * v
    )
    grid_points_3d = grid_points_3d.reshape(-1, 3)  
    
    
    grid_energies = interpolator(grid_points_3d)
    
    
    regularization_term = regularization_coeff * (
        np.linalg.norm(grid_points_3d - prev_point, axis=1) +
        np.linalg.norm(grid_points_3d - next_point, axis=1)
    )
    
    
    total_energies = grid_energies + regularization_term
    
    
    min_energy_idx = np.argmin(total_energies)
    min_energy_point = grid_points_3d[min_energy_idx]
    
    return min_energy_point

def Migration_Energy(potential_file, s_point, e_point, num_points=11, iteration_=10,
                     output="optimized_path.csv"):

    csv_file = potential_file    #"potential.csv"
    df = pd.read_csv(csv_file)

    
    points = df[['x', 'y', 'z']].values  #  (N, 3)
    energy = df['energy'].values  # (N,)

    interpolator = LinearNDInterpolator(points, energy, fill_value=np.max(energy))
    #interpolator = RBFInterpolator(points, energy, kernel='thin_plate_spline')  # 'cubic''thin_plate_spline'

    
    start_point = np.array(s_point)   
    end_point = np.array(e_point)   

    
    num_points = num_points  
    path = np.linspace(start_point, end_point, num_points)


    for iteration in range(100):     
        for i in range(1, num_points - 1):  
            new_point = optimize_point_in_grid(i, path,interpolator , start_point, end_point)
            path[i] = new_point                 
        if iteration > iteration_:
            print(f"Optimization converged after {iteration + 1} iterations.")
            break
    else:
        print("Reached maximum number of iterations without full convergence.")


    # 
    fixed_indices = [0, len(path) // 2, len(path) - 1]
    for iteration in range(100):       #max_iterations
        for i in range(1, num_points - 1):  
            if i not in fixed_indices:  
                path[i] = optimize_point_with_neighbors(i, path, interpolator)    
        
        if iteration > iteration_:
            print(f"Second optimization converged after {iteration + 1} iterations.")
            break
    else:
        print("Reached maximum number of iterations without full convergence in the second optimization.")

    
    output_file = output
    np.savetxt(output_file, path, delimiter=",", header="x,y,z", comments="")
    print(f"Optimized path saved to {output_file}.")

    
    path_energies = interpolator(path)
    energies_min = np.min(path_energies)
    energies_shifted = path_energies - energies_min  

    max_idx = np.argmax(path_energies)
    max_point = path[max_idx]
    print(f"Highest energy point: {max_point}")
    print("E1------ [" + ", ".join(f"{v:.8f}" for v in energies_shifted) + "]")

    return max(energies_shifted), path, energies_shifted


def Migration_Energy_op(path, atom_type, di, outf , POSCAR_rm ,out_xsf, e0):   
    """
    
    """
    path_energies=[]
    trajectory = []
    for points in path:
        e , a= optimize_structure_with_fixed_atom(POSCAR_rm, points, atom_symbol= atom_type, fmax=0.01, steps=300,di=di)
        path_energies.append(e)        
        trajectory.append(a)

    write(out_xsf, trajectory, format="xsf")

    energies_min = np.min(path_energies)
    energies_shifted = path_energies - energies_min
    print("E2------ [" + ", ".join(f"{v:.8f}" for v in energies_shifted) + "]")

    plot_energy_profile_fixed_labels(
        image_indices1 =  np.linspace(0, 1, len(energies_shifted)),
        energies1= e0,
        image_indices2= np.linspace(0, 1, len(energies_shifted)),
        energies2=  energies_shifted ,
        output_filename= outf ,
        save_plot=True,
        show_plot=False
    )

    return  max(energies_shifted)



def potential_draw(energy_b , min_energy,output_potential, output_path , out_html):
    
    df = pd.read_csv(output_potential) 
    df.loc[df['energy'] < (min_energy - 0.1), 'energy'] = min_energy

    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
    energy = df['energy'].values

    
    interpolator = LinearNDInterpolator(points=(x, y, z), values=energy)

    
    grid_x, grid_y, grid_z = np.mgrid[
        x.min():x.max():100j,  
        y.min():y.max():100j,
        z.min():z.max():100j,
    ]

   
    grid_energy = interpolator(grid_x, grid_y, grid_z)

    grid_energy = gaussian_filter(grid_energy, sigma=2)

    
    fig = go.Figure()

    
    fig.add_trace(
        go.Isosurface(
            x=grid_x.ravel(),
            y=grid_y.ravel(),
            z=grid_z.ravel(),
            value=grid_energy.ravel(),
            isomin=np.nanmin(grid_energy)-0.1,  
            isomax=np.nanmin(grid_energy)+ energy_b,  
            surface_count=6,  
            colorscale='Viridis',  
            caps=dict(x_show=False, y_show=False, z_show=False), 
            colorbar_title="Energy", 
            opacity=0.7  
        )
    )

    
    loaded_path = np.loadtxt(output_path, delimiter=",", skiprows=1) 

    
    fig.add_trace(
        go.Scatter3d(
            x=loaded_path[:, 0],  
            y=loaded_path[:, 1],  
            z=loaded_path[:, 2],  
            mode='markers+lines', 
            marker=dict(size=5, color='red'),  
            line=dict(color='blue', width=3),  
            name="Optimized Path"  
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        title="3D Energy Isosurfaces with Smooth Interpolation"
    )

    fig.write_html(out_html)


def plot_energy_profile_fixed_labels(
    image_indices1, energies1,
    image_indices2=None, energies2=None,
    output_filename="Migration_Energy_op.jpg",
    save_plot=True, show_plot=False,
    title=None,
):
    """
    Plot fixed-lattice and (optional) relaxed-lattice migration energy
    profiles with Savitzky-Golay smoothing + cubic-spline interpolation.

    Font family is set to 'sans-serif' only (no specific font name), so that
    the plot renders on headless servers that may not ship Arial/DejaVu.
    Output is written as JPG.
    """
    label1_text = "Fixed"
    color1_val = "#4a90d9"
    marker1_style = "o"
    label2_text = "Optimized"
    color2_val = "#e17c72"
    marker2_style = "s"

    x_label_text = "Path distance"
    y_label_text = "Energy (eV)"
    num_smooth_points_val = 300
    spline_k_val = 3
    figsize_val = (4, 3)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'lines.linewidth': 2.0,
        'lines.markersize': 5,
    })

    fig, ax = plt.subplots(figsize=figsize_val)

    def _plot_one(x, y, color, marker, label):
        x_np = np.asarray(x, dtype=float)
        y_np = np.asarray(y, dtype=float)
        n = len(x_np)

        # Savitzky-Golay filter: window must be odd, >= polyorder+2, <= n.
        y_filt = y_np
        if n >= 5:
            win = min(7, n if n % 2 == 1 else n - 1)
            poly = min(3, win - 2)
            if win >= 3 and poly >= 1:
                try:
                    y_filt = savgol_filter(y_np, win, poly)
                except Exception:
                    y_filt = y_np

        if n > spline_k_val:
            x_smooth = np.linspace(x_np.min(), x_np.max(), num_smooth_points_val)
            try:
                y_smooth = make_interp_spline(x_np, y_filt, k=spline_k_val)(x_smooth)
                ax.plot(x_smooth, y_smooth, color=color, label=label)
            except Exception:
                ax.plot(x_np, y_filt, color=color, label=label)
        else:
            ax.plot(x_np, y_np, color=color, label=label)

        ax.scatter(x_np, y_np, marker=marker,
                   facecolors='none', edgecolors=color, linewidths=1.0)

    _plot_one(image_indices1, energies1, color1_val, marker1_style, label1_text)
    if image_indices2 is not None and energies2 is not None:
        _plot_one(image_indices2, energies2, color2_val, marker2_style, label2_text)

    ax.set_xlabel(x_label_text)
    ax.set_ylabel(y_label_text)
    if title:
        ax.set_title(title)

    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6, color='grey')

    if save_plot and output_filename:
        plt.tight_layout()
        plt.savefig(output_filename, format='jpg', dpi=300, bbox_inches='tight')
        print(f"fig save as: {output_filename}")

    if show_plot:
        plt.tight_layout()
        plt.show()

    plt.close(fig)


def export_path_to_cif(
    structure_file: str,
    hop_list: list,
    migrating_specie: str,
    out_cif: str = "percolation_path.cif",
    add_midpoints: bool = True,
    midpoint_specie: str = "H",
    n_inserts: int = 6,
):
    vis = Structure.from_file(structure_file).copy()
    mids = []
    path_points = []
    prev_fc = None

    for h in hop_list:
        hop = h["hop"]
        is_fc_raw = hop.isite.frac_coords
        es_fc_raw = hop.esite.frac_coords

        if prev_fc is None:
            is_fc = is_fc_raw % 1.0
            path_points.append(is_fc)
        else:
            is_fc = prev_fc

        delta = es_fc_raw - is_fc_raw
        delta = delta - np.round(delta)
        es_fc = is_fc + delta
        path_points.append(es_fc)
        prev_fc = es_fc

        if add_midpoints:
            cart_delta = np.dot(delta, vis.lattice.matrix)
            chord_len = np.linalg.norm(cart_delta)
            if chord_len > 1e-8:
                ref = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(cart_delta, ref) / chord_len) > 0.9:
                    ref = np.array([0.0, 1.0, 0.0])
                n_hat = np.cross(cart_delta, ref)
                n_hat = n_hat / np.linalg.norm(n_hat)
                arc_amp = 0.05 * chord_len
                for kk in range(1, n_inserts + 1):
                    t = kk / (n_inserts + 1)
                    base_fc = is_fc + t * delta
                    base_cart = vis.lattice.get_cartesian_coords(base_fc)
                    offset_cart = arc_amp * np.sin(np.pi * t) * n_hat
                    p_cart = base_cart + offset_cart
                    p_fc = vis.lattice.get_fractional_coords(p_cart) % 1.0
                    mids.append(p_fc)

    for fc in path_points:
        vis.append(migrating_specie, fc % 1.0)
    for fc in mids:
        vis.append(midpoint_specie, fc)

    vis.to(out_cif)
    print(f"Saved CIF: {out_cif}")