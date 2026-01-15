import numpy as np
import itertools

def generate_spacetree(element_indices, max_depth, n_seed, mesh_now, domain):
    begin, end, depth, partitions = 0, 1, 0, []
    tree = [((-1.0, 1.0),)]
    for _ in range(mesh_now.dim - 1):
        tree[0] += ((-1.0, 1.0),)
    
    while begin != end and depth < max_depth:
        for current_partition in tree[begin:end]:

            seeds = []
            for i_d in range(mesh_now.dim):
                coords_dummy_start, coords_dummy_end = np.zeros(mesh_now.dim), np.zeros(mesh_now.dim)
                coords_dummy_start[i_d] = current_partition[i_d][0]
                coords_dummy_end[i_d] = current_partition[i_d][1]
                seed = np.linspace(mesh_now.map_to_physical(element_indices, coords_dummy_start)[i_d],
                                   mesh_now.map_to_physical(element_indices, coords_dummy_end)[i_d], n_seed)
                seeds.append(seed)
            coords_grid = np.meshgrid(*seeds, indexing='ij')
            coords_list = [coord.ravel() for coord in coords_grid]
            result = domain(coords_list)

            # Subdivide if some are inside and some outside
            if not np.all(result) and not np.all(np.logical_not(result)):
                # create subdivisions                
                new_partitions = []
                for half_combo in itertools.product(*[(0, 1)] * mesh_now.dim):
                    new_part = []
                    for d in range(mesh_now.dim):
                        min_d, max_d = current_partition[d]
                        mid_d = (min_d + max_d) / 2
                        if half_combo[d] == 0:
                            new_part.append((min_d, mid_d))  # Lower half
                        else:
                            new_part.append((mid_d, max_d))  # Upper half
                    new_partitions.append(tuple(new_part))
                tree += new_partitions
            else:
                partitions.append(current_partition)
        begin, end, depth = end, len(tree), depth + 1

    return partitions + tree[begin:end]
