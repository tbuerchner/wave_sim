import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import itertools
from tqdm import tqdm

import material
import mesh_cg as mesh
import integration
import ansatz
import functions
import geometries
import spacetree

# Spatial dimension
dim = 2
print("Setting up 2D SEM problem...")

# domain
origin = np.array([0.0, 0.0])
length = np.array([2.0, 2.0])
print(f"Domain origin: {origin}, length: {length}")

# mesh
n_elements = np.array([20, 20])
polynomial_degree = np.array([4, 4])
print(f"Number of elements: {n_elements}, Polynomial degree: {polynomial_degree}")
mesh_2d = mesh.ndRectangle(dim=dim, origin=origin, lengths=length, n_elements=n_elements, polynomial_degree=polynomial_degree)

# physical domain (plate with a hole)
center = np.array([length[0], length[1]/2])
radius = 0.1
# domain = geometries.hyperrectangle_with_hypersphere(center=center, radius=radius)
domain = geometries.all_physical(center=center, radius=radius)

# stabilization parameters for cut cells and parameter for spacetree
alpha = 1e-2
depth = np.max(polynomial_degree) + 1
n_seed = 10

# material 
density = 1.0
wave_speed = 1.0
print(f"Material properties - Density: {density}, Wave Speed: {wave_speed}")
mat = material.acousticMaterial(density=density, wave_speed=wave_speed)

# time
T = length[0] / 2 / wave_speed # 2 * length[0] / wave_speed
n_steps = 2000
print(f"Total simulation time: {T}, Number of time steps: {n_steps}")
dt = T / n_steps

# initial conditions
frequency = 2.0
x0 = 0.0
sigma = wave_speed / frequency / 20
print(f"Initial condition - Gaussian pulse centered at {x0} with sigma {sigma}")
ic_func = functions.gaussian_pulse_1d(x0=x0, sigma=sigma, wave_speed=wave_speed)
def ic_0_func(coords_physical):
    return ic_func(coords_physical[0], 0.0)
def ic_1_func(coords_physical):
    return ic_func(coords_physical[0], dt)


# external force
def f_ex_func(coords_physical): 
    return 0.0
def f_t(t):
    return 0.0

# defining shape functions
gll_nodes = [integration.get_gll_integration(polynomial_degree[i_d] + 1)[0] for i_d in range(dim)]
shapes_0 = [ansatz.get_lagrange_functions(gll_nodes[i_d], derivative_order=0) for i_d in range(dim)]
shapes_1 = [ansatz.get_lagrange_functions(gll_nodes[i_d], derivative_order=1) for i_d in range(dim)]

# integration points and weights
int_points_rule1, int_weights_rule1 = [], []
int_points_rule2, int_weights_rule2 = [], []
for i_d in range(dim):
    int_p, int_w = integration.get_gll_integration(polynomial_degree[i_d] + 1)
    int_points_rule1.append(int_p)
    int_weights_rule1.append(int_w)
    int_p, int_w = np.polynomial.legendre.leggauss(polynomial_degree[i_d] + 1)
    int_points_rule2.append(int_p)
    int_weights_rule2.append(int_w)

print("2D SEM problem setup complete.")

print("Starting with assembly of the global matrices...")

# data arrays global matrices in COO format
n_entries_assembly = mesh_2d.n_elements_total * mesh_2d.n_dofs_element ** 2
row = np.zeros(n_entries_assembly, dtype=np.uint)
col = np.zeros(n_entries_assembly, dtype=np.uint)
data_M = np.zeros(n_entries_assembly)
data_K = np.zeros(n_entries_assembly)
data_proj = np.zeros(n_entries_assembly)

# data arrays for global external force and initial conditions
f_ex = np.zeros(mesh_2d.n_dofs_global)
ic_0 = np.zeros(mesh_2d.n_dofs_global)
ic_1 = np.zeros(mesh_2d.n_dofs_global)


for element_indices in tqdm(itertools.product(*[range(n) for n in np.flip(n_elements)]), 
                     total=mesh_2d.n_elements_total, desc="Processing elements"):
    element_indices = element_indices[::-1]
    tqdm.write(f"Current element indices: {element_indices}")

    # Local element matrices 
    proje = np.zeros((mesh_2d.n_dofs_element, mesh_2d.n_dofs_element))
    Me = np.zeros((mesh_2d.n_dofs_element, mesh_2d.n_dofs_element))
    Ke = np.zeros((mesh_2d.n_dofs_element, mesh_2d.n_dofs_element))

    # Local element vectors
    f_exe = np.zeros((mesh_2d.n_dofs_element,))
    ic_0e = np.zeros((mesh_2d.n_dofs_element,))
    ic_m1e = np.zeros((mesh_2d.n_dofs_element,))

    # generate spacetree for the current element
    partitions = spacetree.generate_spacetree(element_indices, depth, n_seed, mesh_2d, domain)
    cell_is_cut = len(partitions) > 1

    # loop over partitions of the spacetree
    for part in partitions:
        # integrate over whole element 
        # (structure expandable for higher dimensions and cut cells)
        r_values_rule1 = [[xi * (part[i_d][1] - part[i_d][0]) / 2 + (part[i_d][1] + part[i_d][0]) / 2 
                        for xi in int_points_rule1[i_d]] for i_d in range(dim)]
        r_values_rule1 = [[xi * (part[i_d][1] - part[i_d][0]) / 2 + (part[i_d][1] + part[i_d][0]) / 2 
                        for xi in int_points_rule1[i_d]] for i_d in range(dim)]
        r_values_rule2 = [[xi * (part[i_d][1] - part[i_d][0]) / 2 + (part[i_d][1] + part[i_d][0]) / 2 
                       for xi in int_points_rule2[i_d]] for i_d in range(dim)]

        # Evaluate shape functions and their derivatives at integration points
        # rule 1 for mass: just 0th derivative neeed
        # rule 2 for everything else: 0th and 1st derivative needed
        shapes_r_rule1_0 = [[np.array([shape(r) for shape in shapes_0[i_d]]) 
                            for r in r_values_rule1[i_d]] for i_d in range(dim)]
        shapes_r_rule2_0 = [[np.array([shape(r) for shape in shapes_0[i_d]]) 
                        for r in r_values_rule2[i_d]] for i_d in range(dim)]
        shapes_r_rule2_1 = [[np.array([shape(r) for shape in shapes_1[i_d]]) 
                        for r in r_values_rule2[i_d]] for i_d in range(dim)]
        
        for integration_indices in itertools.product(*[range(n) for n in np.flip(polynomial_degree + 1)]):
            igauss = integration_indices[::-1]

            # scale weight for physical element
            weight1 = 1.0
            ip_local1=[]
            for i_d in range(dim):
                weight1 *= int_weights_rule1[i_d][igauss[i_d]] * mesh_2d.dx[i_d] / 2 * (part[i_d][1] - part[i_d][0]) / 2
                ip_local1.append(r_values_rule1[i_d][igauss[i_d]])

            if domain(mesh_2d.map_to_physical(element_indices, ip_local1)) < 0.5:
                weight1 *= alpha
            # Add contributions
            N = np.outer(shapes_r_rule1_0[1][igauss[1]], shapes_r_rule1_0[0][igauss[0]]).ravel()
            Me += weight1 * mat.density * np.outer(N, N)

             # scale weight for physical element
            weight2 = 1.0
            ip_local2=[]
            for i_d in range(dim):
                weight2 *= int_weights_rule2[i_d][igauss[i_d]] * mesh_2d.dx[i_d] / 2 * (part[i_d][1] - part[i_d][0]) / 2
                ip_local2.append(r_values_rule2[i_d][igauss[i_d]])

            if domain(mesh_2d.map_to_physical(element_indices, ip_local2)) < 0.5:
                weight2 *= alpha
            # Add contributions
            N0 = np.outer(shapes_r_rule2_0[1][igauss[1]], shapes_r_rule2_0[0][igauss[0]]).ravel()
            N1x = np.outer(shapes_r_rule2_0[1][igauss[1]], shapes_r_rule2_1[0][igauss[0]] * (2.0 / mesh_2d.dx[0])).ravel()  # chain rule for derivative
            N1y = np.outer(shapes_r_rule2_1[1][igauss[1]] * (2.0 / mesh_2d.dx[1]), shapes_r_rule2_0[0][igauss[0]]).ravel()  # chain rule for derivative
            
            Ke += weight2 * mat.density * mat.wave_speed ** 2 * (np.outer(N1x, N1x) + np.outer(N1y, N1y))
            proje += weight2 * np.outer(N0, N0)
            x_phys = mesh_2d.map_to_physical(element_indices, [r_values_rule2[0][igauss[0]], r_values_rule2[1][igauss[1]]])
            f_exe += weight2 * f_ex_func(x_phys) * N0
            ic_0e += weight2 * ic_0_func(x_phys) * N0
            ic_m1e += weight2 * ic_1_func(x_phys) * N0

    # get the location map for the current element
    lm = mesh_2d.get_location_map(element_indices)
    # Repeat locationMaps as colums/rows linearize the result to obtain matrix entry coordinates
    element_slice = mesh_2d.get_element_slice(element_indices)

    row[element_slice] = np.broadcast_to(lm, (mesh_2d.n_dofs_element, mesh_2d.n_dofs_element)).ravel()
    col[element_slice] = np.broadcast_to(lm, (mesh_2d.n_dofs_element, mesh_2d.n_dofs_element)).T.ravel()

    data_M[element_slice] = Me.ravel()
    data_K[element_slice] = Ke.ravel()
    data_proj[element_slice] = proje.ravel()
    f_ex[lm] += f_exe
    ic_0[lm] += ic_0e
    ic_1[lm] += ic_m1e

# Create coordinate matrix and convert to compressed sparse column format
M = scipy.sparse.coo_matrix((data_M, (row, col)), shape=(mesh_2d.n_dofs_global, mesh_2d.n_dofs_global)).tocsc()
K = scipy.sparse.coo_matrix((data_K, (row, col)), shape=(mesh_2d.n_dofs_global, mesh_2d.n_dofs_global)).tocsc()
proj = scipy.sparse.coo_matrix((data_proj, (row, col)), shape=(mesh_2d.n_dofs_global, mesh_2d.n_dofs_global)).tocsc()

# for boundary-conforming SEM we can take the diagonal of the mass matrix and invert it directly
print("Average bandwidth before filtering: " + "%.2f" % (M.nnz / M.shape[0]))
M.data[np.abs(M.data) < 1e-16 * scipy.sparse.linalg.norm(M)] = 0.0
M.eliminate_zeros()
print("Average bandwidth after filtering: " + "%.2f" % (M.nnz / M.shape[0]))

print("Completed assembly of the global matrices...")

print("Time integration (" + str(n_steps) + " time steps) ... ")

# Sparse LU factorization of and time integration
factorized_M = scipy.sparse.linalg.splu(M)
factorized_proj = scipy.sparse.linalg.splu(proj)

u = np.zeros((n_steps + 1, mesh_2d.n_dofs_global))
u[0] = factorized_proj.solve(ic_0)
u[1] = factorized_proj.solve(ic_1)

for i_step in tqdm(range(1, n_steps), desc="Time stepping"):
    # compute internal + external forces
    rhs = dt ** 2 *(- K.dot(u[i_step]) + f_ex * f_t(i_step * dt))
    u[i_step + 1] = 2.0 * u[i_step] - u[i_step - 1] + factorized_M.solve(rhs)
    
print("Time integration complete.")

# animation of all time steps
print("Start animation of the results.")
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(origin[0], origin[0] + length[0])
ax.set_ylim(origin[1], origin[1] + length[1])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('2D SEM Wave Propagation')
ax.set_aspect('equal', adjustable='box')
def init():
    line.set_data([], [])
    return line,
x = np.empty(mesh_2d.n_dofs_global_directions[0])
y = np.empty(mesh_2d.n_dofs_global_directions[1])
for i_el_x in range(n_elements[0]):
    for i_p_x, point_x in enumerate(gll_nodes[0]):
        x[i_el_x * (polynomial_degree[0]) + i_p_x] = mesh_2d.map_to_physical([i_el_x, 0], [point_x, 0.0])[0]
for i_el_y in range(n_elements[1]):
    for i_p_y, point_y in enumerate(gll_nodes[1]):
        y[i_el_y * (polynomial_degree[1]) + i_p_y] = mesh_2d.map_to_physical([0, i_el_y], [0.0, point_y])[1]
X, Y = np.meshgrid(x, y, indexing='ij')
Xe, Ye = np.meshgrid(np.linspace(0, length[0], n_elements[0] + 1),
                     np.linspace(0, length[1], n_elements[1] + 1), indexing='ij')

# Z = np.reshape(u[-1], (mesh_2d.n_dofs_global_directions[1], mesh_2d.n_dofs_global_directions[0])).T
# ax.clear()
# ax.plot(Xe, Ye, color='lightgray', linewidth=1.0)  
# ax.plot(Xe.T, Ye.T, color='lightgray', linewidth=1.0) 
# ax.contourf(X, Y, Z, levels=np.linspace(-.5, .5, 48), cmap='seismic', extend='both')

# plt.show()

# quit()

def animate(i):
    Z = np.reshape(u[i], (mesh_2d.n_dofs_global_directions[1], mesh_2d.n_dofs_global_directions[0])).T
    ax.clear()
    ax.plot(Xe, Ye, color='lightgray', linewidth=1.0)  
    ax.plot(Xe.T, Ye.T, color='lightgray', linewidth=1.0) 
    ax.contourf(X, Y, Z, levels=np.linspace(-1.0, 1.0, 48), cmap='seismic', extend='both')
    return ax,
ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=range(0, n_steps+1, 20), interval=1, blit=True)
plt.show()

