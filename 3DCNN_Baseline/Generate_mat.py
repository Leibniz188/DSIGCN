"""
Inclusion Composites - .mat Data Generator
Directly generates 3D binary phase data for CNN training.

1. Read node.text -> Get node coordinates and determine boundaries.
2. Read element.text -> Get all tetrahedral elements.
3. Read matrix.txt -> Get matrix element IDs.
4. Calculate inclusions = All elements - Matrix elements.
5. Generate a 101³ lattice grid (initialized to 0).
6. Check if each grid point lies inside any inclusion tetrahedron.
7. Mark points inside inclusions as 1.
8. Output as .mat format.
"""

import os
import time
import argparse
import scipy.io
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


class MatGenerator:
    """Generator for voxelizing  inclusion meshes into 3D grids."""

    def __init__(self, lattice_size=101, device='cuda'):
        self.lattice_size = lattice_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        print(f"Device: {self.device}")
        print(f"Lattice: {lattice_size}³ = {lattice_size**3:,} voxels")

    def read_node_file(self, filepath):
        """Reads node.txt file to capture 3D node coordinates."""
        nodes = {}
        with open(filepath, 'r') as f:
            _ = f.readline()  # Skip the first line containing node counts
            for line in f:
                line = line.strip().rstrip(',')
                if not line:
                    continue
                parts = [p.strip() for p in line.split(',') if p.strip()]
                if len(parts) >= 4:
                    node_id = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    nodes[node_id] = np.array([x, y, z])
        return nodes

    def read_element_file(self, filepath):
        """Reads element.txt file containing tetrahedral element configurations."""
        elements = []
        with open(filepath, 'r') as f:
            _ = f.readline()  # Skip the first line containing element counts
            for line in f:
                line = line.strip().rstrip(',')
                if not line:
                    continue
                parts = [p.strip() for p in line.split(',') if p.strip()]
                if len(parts) >= 5:  # Element ID followed by 4 node IDs
                    elem_id = int(parts[0])
                    node_ids = [int(parts[i]) for i in range(1, 5)]
                    elements.append((elem_id, node_ids))
        return elements

    def read_matrix_file(self, filepath):
        """Reads matrix.txt file containing IDs belonging to the matrix phase."""
        matrix_ids = set()
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                for part in parts:
                    if part.isdigit():
                        matrix_ids.add(int(part))
        return matrix_ids

    def generate_lattice(self, bounds):
        """Generates a uniform 3D grid point array within the mesh bounds."""
        x = torch.linspace(bounds[0, 0], bounds[0, 1], self.lattice_size, dtype=self.dtype, device=self.device)
        y = torch.linspace(bounds[1, 0], bounds[1, 1], self.lattice_size, dtype=self.dtype, device=self.device)
        z = torch.linspace(bounds[2, 0], bounds[2, 1], self.lattice_size, dtype=self.dtype, device=self.device)

        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)
        return points

    def check_points_in_tetrahedra(self, points, tetrahedra, chunk_size=100):
        """GPU-accelerated batch verification of points located inside tetrahedra."""
        N = points.shape[0]
        M = tetrahedra.shape[0]

        if M == 0:
            return torch.zeros(N, dtype=torch.bool, device=self.device)

        result = torch.zeros(N, dtype=torch.bool, device=self.device)

        # Precompute bounding boxes for filtering acceleration
        tet_min = tetrahedra.min(dim=1)[0]
        tet_max = tetrahedra.max(dim=1)[0]

        # Chunk processing loop to manage GPU memory usage
        for chunk_start in range(0, M, chunk_size):
            chunk_end = min(chunk_start + chunk_size, M)
            tet_chunk = tetrahedra[chunk_start:chunk_end]
            tet_min_chunk = tet_min[chunk_start:chunk_end]
            tet_max_chunk = tet_max[chunk_start:chunk_end]

            # Broad-phase bounding box filter
            points_exp = points.unsqueeze(1)
            in_bbox = ((points_exp >= tet_min_chunk.unsqueeze(0)) &
                       (points_exp <= tet_max_chunk.unsqueeze(0))).all(dim=2)

            point_indices, tet_indices = torch.where(in_bbox)

            if len(point_indices) == 0:
                continue

            pts = points[point_indices]
            tets = tet_chunk[tet_indices]

            v0, v1, v2, v3 = tets[:, 0, :], tets[:, 1, :], tets[:, 2, :], tets[:, 3, :]
            is_inside = self._barycentric_check(pts, v0, v1, v2, v3)
            result[point_indices[is_inside]] = True

        return result

    def _barycentric_check(self, pts, v0, v1, v2, v3):
        """Narrow-phase check using Barycentric coordinates calculation."""
        mat = torch.stack([v1 - v0, v2 - v0, v3 - v0], dim=2)
        det = torch.linalg.det(mat)

        valid_mask = torch.abs(det) > 1e-10
        if not valid_mask.any():
            return torch.zeros(pts.shape[0], dtype=torch.bool, device=self.device)

        pts_valid = pts[valid_mask]
        v0_valid = v0[valid_mask]
        mat_valid = mat[valid_mask]

        diff = pts_valid - v0_valid

        try:
            lambdas = torch.linalg.solve(mat_valid, diff.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            return torch.zeros(pts.shape[0], dtype=torch.bool, device=self.device)

        lambda0 = 1.0 - lambdas.sum(dim=1)

        eps = 1e-6
        inside_valid = ((lambda0 >= -eps) & (lambda0 <= 1 + eps) &
                        (lambdas[:, 0] >= -eps) & (lambdas[:, 0] <= 1 + eps) &
                        (lambdas[:, 1] >= -eps) & (lambdas[:, 1] <= 1 + eps) &
                        (lambdas[:, 2] >= -eps) & (lambdas[:, 2] <= 1 + eps))

        result = torch.zeros(pts.shape[0], dtype=torch.bool, device=self.device)
        result[valid_mask] = inside_valid

        return result

    def process_single_sample(self, sample_dir):
        """Extracts mesh data and constructs a single 3D voxel grid array."""
        sample_path = Path(sample_dir)

        node_file = sample_path / 'node.txt'
        element_file = sample_path / 'element.txt'
        matrix_file = sample_path / 'matrix.txt'

        # Safety fallback check for missing files
        if not all([node_file.exists(), element_file.exists(), matrix_file.exists()]):
            print(f"  ⚠ Missing mesh configuration files. Returning empty array.")
            return np.zeros((self.lattice_size, self.lattice_size, self.lattice_size), dtype=np.float32)

        # Loading raw structures
        nodes = self.read_node_file(node_file)
        all_elements = self.read_element_file(element_file)
        matrix_ids = self.read_matrix_file(matrix_file)

        # Isolate inclusion elements (elements omitted from matrix.txt)
        inclusion_elements = [(elem_id, node_ids) for elem_id, node_ids in all_elements
                              if elem_id not in matrix_ids]

        if len(inclusion_elements) == 0:
            print(f"  ⚠ No inclusion elements identified. Returning all-zero array.")
            return np.zeros((self.lattice_size, self.lattice_size, self.lattice_size), dtype=np.float32)

        # Assemble localized coordinate arrays
        node_ids = sorted(nodes.keys())
        node_coords = np.array([nodes[nid] for nid in node_ids])

        # Track external mesh boundaries
        bounds = np.column_stack([node_coords.min(axis=0), node_coords.max(axis=0)])

        # Map lattice space
        points = self.generate_lattice(bounds)

        # Resolve targeted volumetric coordinate points 
        inclusion_tets = []
        for elem_id, node_ids_list in inclusion_elements:
            try:
                tet_coords = np.array([nodes[nid] for nid in node_ids_list])
                inclusion_tets.append(tet_coords)
            except KeyError:
                continue

        if len(inclusion_tets) == 0:
            print(f"  ⚠ No valid inclusion tetrahedra found.")
            return np.zeros((self.lattice_size, self.lattice_size, self.lattice_size), dtype=np.float32)

        tetrahedra = torch.from_numpy(np.array(inclusion_tets)).to(dtype=self.dtype, device=self.device)

        # Calculate logical intersections
        inside_mask = self.check_points_in_tetrahedra(points, tetrahedra, chunk_size=100)

        # Reshape flat labels to a standard 3D voxel grid
        labels = inside_mask.cpu().numpy().astype(np.float32)
        voxel_grid = labels.reshape(self.lattice_size, self.lattice_size, self.lattice_size)

        return voxel_grid

    def generate_mat_files(self, base_dir, output_dir, start_idx=1, end_idx=1000, samples_per_file=200):
        """Processes files incrementally and outputs batched MATLAB .mat matrices."""
        base_path = Path(base_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Inclusion Data Voxelization")
        print(f"{'='*70}")
        print(f"Input Directory:  {base_dir}")
        print(f"Output Directory: {output_dir}")
        print(f"Sample Range:     {start_idx} -> {end_idx}")
        print(f"Samples/File:     {samples_per_file}")
        print(f"{'='*70}\n")

        start_time = time.time()

        # Batch grouping iteration
        for batch_start in range(start_idx, end_idx + 1, samples_per_file):
            batch_end = min(batch_start + samples_per_file - 1, end_idx)

            print(f"\nProcessing Batch: Samples {batch_start}-{batch_end}")
            phase_data = []

            for i in tqdm(range(batch_start, batch_end + 1), desc="  Generating Voxels"):
                sample_dir = next(base_path.glob(f'*{i}_mesh'))

                if not sample_dir.exists():
                    print(f"    ⚠ Directory not found: {sample_dir.name}, skipping with empty array.")
                    voxel_grid = np.zeros((self.lattice_size, self.lattice_size, self.lattice_size), dtype=np.float32)
                else:
                    try:
                        voxel_grid = self.process_single_sample(sample_dir)
                    except Exception as e:
                        print(f"    ✗ Sample {i} processing failure: {e}, fallback to empty array.")
                        voxel_grid = np.zeros((self.lattice_size, self.lattice_size, self.lattice_size), dtype=np.float32)

                phase_data.append(voxel_grid)

                # Regularly wipe CUDA cache to avoid memory fragmentation
                if i % 10 == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            # Compile matrix stack into target compressed output file structures
            phase_array = np.array(phase_data, dtype=np.float32)
            batch_name = f'Voxel_{batch_start-1}_{batch_end-1}'
            output_file = output_path / f'{batch_name}.mat'

            scipy.io.savemat(str(output_file), {'phase': phase_array})

            print(f"  ✓ File Saved: {output_file.name}")
            print(f"    Shape: {phase_array.shape}")
            print(f"    Inclusion Volume Fraction: {phase_array.mean():.3f}")

        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Execution Completed Successfully!")
        print(f"Total Time: {total_time/60:.2f} minutes")
        print(f"Average Processing Speed: {total_time/(end_idx-start_idx+1):.2f} seconds/sample")
        print(f"{'='*70}\n")


def main():
    """Main execution block containing parameter configurations."""
    # Dynamically locate base path relative to this script's position

    parser = argparse.ArgumentParser(description='Convert inclusion mesh data into voxel grid matrices.')
    
    # ─── PERFECTLY CONNECTED TO THE EXTRACTED FOLDER ───
    parser.add_argument('--base_dir', type=str,
                        default=os.path.join('../example_data/copped_fiber/extracted/'),
                        help='Directory containing the extracted "XXX-X_mesh" folders (node.txt, element.txt, matrix.txt).')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join('../example_data/copped_fiber/output_mat/'),
                        help='Directory paths designated to save final output .mat collections.')
    # ───────────────────────────────────────────────────
    
    parser.add_argument('--start_idx', type=int, default=1,
                        help='Starting sample parsing range index.')
    parser.add_argument('--end_idx', type=int, default=10, # Adjust according to the size of the dataset.
                        help='Terminal boundary sample index.')
    parser.add_argument('--samples_per_file', type=int, default=10,
                        help='Number of parsed voxel configurations bundled into each unique .mat stack file.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computational execution backend interface resource target hardware engine (cuda/cpu).')

    args = parser.parse_args()

    generator = MatGenerator(
        lattice_size=101,
        device=args.device
    )

    generator.generate_mat_files(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        samples_per_file=args.samples_per_file
    )


if __name__ == '__main__':
    main()