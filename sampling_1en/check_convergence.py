import numpy as np
import matplotlib.pyplot as plt
from h5 import HDFArchive
from triqs.gf import *

# max_G_diff 関数をここで定義 (元のコードを利用)
def max_G_diff(G1, G2, norm_temp=True):
    if isinstance(G1, BlockGf):
        diff = 0.0
        for block, gf in G1:
            diff += max_G_diff(G1[block], G2[block], norm_temp)
        return diff

    assert G1.mesh == G2.mesh, 'mesh of two input Gfs does not match'
    assert G1.target_shape == G2.target_shape, 'can only compare Gfs with same shape'

    if type(G1.mesh) is MeshImFreq:
        offset = np.diag(np.diag(G1.data[-1,:,:].real - G2.data[-1,:,:].real))
    else:
        offset = 0.0

    norm_grid = abs(np.linalg.norm(G1.data - G2.data - offset, axis=tuple(range(1, G1.data.ndim))))
    norm = np.linalg.norm(norm_grid, axis=0)

    if type(G1.mesh) is MeshImFreq:
        norm = np.linalg.norm(norm_grid, axis=0) / np.sqrt(G1.mesh.beta)
    elif type(G1.mesh) is MeshImTime:
        norm = np.linalg.norm(norm_grid, axis=0) * np.sqrt(G1.mesh.beta / len(G1.mesh))
    elif type(G1.mesh) is MeshReFreq:
        norm = np.linalg.norm(norm_grid, axis=0) / np.sqrt(len(G1.mesh))
    else:
        raise ValueError('MeshReTime is not implemented')

    if type(G1.mesh) in (MeshImFreq, MeshImTime) and norm_temp:
        norm = norm / np.sqrt(G1.mesh.beta)

    return norm

# HDF5ファイルのパスを指定
filename = "nsp_bk.h5"

# プロットの準備 (sharex=Trueを追加して横軸を共有)
fig, axs = plt.subplots(5, 1, figsize=(8, 15), sharex=True)

# 化学ポテンシャルのプロット
with HDFArchive(filename, 'r') as ar:
    dmft_results = ar["DMFT_results"]
    chemical_potentials = []
    iteration_numbers = []

    if "Iterations" in dmft_results:
        iterations_group = dmft_results["Iterations"]
        for it in range(len(iterations_group)):
            iteration_key = f"chemical_potential{it}"
            if iteration_key in iterations_group:
                chemical_potential = iterations_group[iteration_key]
                chemical_potentials.append(chemical_potential)
                iteration_numbers.append(it)

    axs[0].plot(iteration_numbers, chemical_potentials, marker='o')
    axs[0].set_ylabel("Chemical Potential")
    axs[0].set_title("Chemical Potential vs Iteration Number")
    axs[0].grid(True)
print(f"delta mu:{chemical_potentials[-1] - chemical_potentials[-2]}")

# GimpとGlocの差のプロット
ar = HDFArchive(filename, 'r')
n_iterations = ar["DMFT_results"]["iteration_count"]

G_diff = []
for it in range(n_iterations + 1):
    Gimp_it = ar["DMFT_results"]["Iterations"]["Gimp_it" + str(it)]
    Gloc_it = ar["DMFT_results"]["Iterations"]["Gloc_it" + str(it)]
    
    diff = max_G_diff(Gimp_it, Gloc_it)
    G_diff.append(diff)

iterations = range(n_iterations + 1)
axs[1].plot(iterations, G_diff, marker='o')
axs[1].set_ylabel("Diff between Gloc and Gimp")
axs[1].set_title("Difference between Gloc and Gimp over iterations")
axs[1].grid(True)
axs[1].set_yscale('log')
print(f"G_loc-Gimp:{G_diff[-1]}")

# Gimpのイテレーションごとの変化のプロット
Gimp_diff = []
for it in range(1, n_iterations + 1):
    Gimp_it = ar["DMFT_results"]["Iterations"]["Gimp_it" + str(it)]
    Gimp_prev = ar["DMFT_results"]["Iterations"]["Gimp_it" + str(it - 1)]
    
    diff = max_G_diff(Gimp_it, Gimp_prev)
    Gimp_diff.append(diff)

iterations = range(1, n_iterations + 1)
axs[2].plot(iterations, Gimp_diff, marker='o')
axs[2].set_ylabel("Diff between Gimp(n) and Gimp(n-1)")
axs[2].set_title("Difference between Gimp(n) and Gimp(n-1) over iterations")
axs[2].grid(True)
axs[2].set_yscale('log')
print(f"G_imp_diff:{Gimp_diff[-1]}")

# Density matrixのプロット
density_matrix_values = []
for it in range(n_iterations):
    density_matrix_it = ar["DMFT_results"]["Iterations"]["Gloc_it" + str(it)].density()
    
    total_real_sum = sum(float(np.real(density_matrix_it[key])) for key in density_matrix_it)
    density_matrix_values.append(total_real_sum)

iterations = range(1, n_iterations + 1)
axs[3].plot(iterations, density_matrix_values, marker='o')
axs[3].set_ylabel("Density Matrix Value")
axs[3].set_title("Density Matrix over iterations")
axs[3].grid(True)
print(f"delta dm:{density_matrix_values[-1]- density_matrix_values[-2]}")

# Orbital resolved density matrixのプロット
orbital_resolved_dms = []
for it in range(n_iterations):
    orbital_resolved_dm_it = ar["DMFT_results"]["Iterations"]["density_matrix_it" + str(it)]
    orbital_resolved_dms.append(orbital_resolved_dm_it)

real_density_data = {}
for data in orbital_resolved_dms:
    for key in data:
        if key not in real_density_data:
            real_density_data[key] = []
        real_density_data[key].append(float(np.real(data[key])))

iterations = range(1, n_iterations + 1)
for key, values in real_density_data.items():
    axs[4].plot(iterations, values, label=key, marker='o')

axs[4].set_xlabel('Iteration')
axs[4].set_ylabel('Orbital Resolved Density (Real Part)')
axs[4].set_title('Orbital Resolved Density Matrix over Iterations')
axs[4].grid(True)
axs[4].legend()
for key, values in real_density_data.items():
    print(f"delta dm orb:{values[-1] - values[-2]}")

# プロットを表示
plt.tight_layout()
plt.show()

plt.savefig("convergence.pdf")