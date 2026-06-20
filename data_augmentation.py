"""
RVE Data Augmentation Module
Features:
1. Geometric rotation (90°/180°/270° around each coordinate axis)
2. Accurate transformation of elastic constants (Engineering Constants -> Compliance Matrix -> Stiffness Matrix -> Rotation -> Engineering Constants)
"""

import numpy as np
import torch
import dgl
from copy import deepcopy


# ==============================================================================
# 1. Rotation Matrix Generation
# ==============================================================================
def get_rotation_matrix(axis, angle_deg):
    """
    Generate a rotation matrix.

    Args:
        axis: 'x', 'y', or 'z'
        angle_deg: Rotation angle in degrees, typically 90, 180, or 270

    Returns:
        R: 3x3 rotation matrix
    """
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)

    if axis == 'x':
        R = np.array([
            [1,  0,  0],
            [0,  c, -s],
            [0,  s,  c]
        ])
    elif axis == 'y':
        R = np.array([
            [ c,  0,  s],
            [ 0,  1,  0],
            [-s,  0,  c]
        ])
    elif axis == 'z':
        R = np.array([
            [ c, -s,  0],
            [ s,  c,  0],
            [ 0,  0,  1]
        ])
    else:
        raise ValueError(f"Invalid axis: {axis}")

    return R.astype(np.float32)


# ==============================================================================
# 2. Elastic Constant Transformations (Core)
# ==============================================================================
def engineering_to_compliance(E11, E22, E33, G23, G13, G12,
                              nu12, nu13, nu21, nu23, nu31, nu32):
    """
    Engineering Constants -> Compliance Matrix S (6x6)

    Voigt notation: [11, 22, 33, 23, 13, 12]

    Compliance Matrix S:
    [ε11]   [S11 S12 S13  0   0   0 ] [σ11]
    [ε22]   [S21 S22 S23  0   0   0 ] [σ22]
    [ε33] = [S31 S32 S33  0   0   0 ] [σ33]
    [γ23]   [ 0   0   0  S44  0   0 ] [τ23]
    [γ13]   [ 0   0   0   0  S55  0 ] [τ13]
    [γ12]   [ 0   0   0   0   0  S66] [τ12]

    Where:
    S11 = 1/E11,  S22 = 1/E22,  S33 = 1/E33
    S12 = -nu12/E11 = -nu21/E22  (Symmetry)
    S13 = -nu13/E11 = -nu31/E33
    S23 = -nu23/E22 = -nu32/E33
    S44 = 1/G23,  S55 = 1/G13,  S66 = 1/G12
    """
    S = np.zeros((6, 6), dtype=np.float32)

    # Diagonal elements
    S[0, 0] = 1.0 / E11
    S[1, 1] = 1.0 / E22
    S[2, 2] = 1.0 / E33
    S[3, 3] = 1.0 / G23
    S[4, 4] = 1.0 / G13
    S[5, 5] = 1.0 / G12

    # Off-diagonal elements (Poisson's ratios)
    # Using symmetry: nu_ij/E_i = nu_ji/E_j
    S[0, 1] = S[1, 0] = -nu12 / E11  # or -nu21/E22
    S[0, 2] = S[2, 0] = -nu13 / E11  # or -nu31/E33
    S[1, 2] = S[2, 1] = -nu23 / E22  # or -nu32/E33

    return S


def compliance_to_stiffness(S):
    """
    Compliance Matrix S -> Stiffness Matrix C
    C = inv(S)
    """
    try:
        C = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        # If singular, use pseudo-inverse
        C = np.linalg.pinv(S)

    return C.astype(np.float32)


def voigt_to_tensor(C_voigt):
    """
    Convert Voigt notation (6x6) to a full 4th-order tensor (3x3x3x3).

    Voigt notation mapping:
    11->0, 22->1, 33->2, 23->3, 13->4, 12->5
    """
    C_tensor = np.zeros((3, 3, 3, 3), dtype=np.float32)

    # Voigt index mapping
    voigt_map = [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1)]

    for I in range(6):
        i, j = voigt_map[I]
        for J in range(6):
            k, l = voigt_map[J]

            # Account for symmetry
            C_tensor[i,j,k,l] = C_voigt[I,J]
            C_tensor[j,i,k,l] = C_voigt[I,J]
            C_tensor[i,j,l,k] = C_voigt[I,J]
            C_tensor[j,i,l,k] = C_voigt[I,J]

    return C_tensor


def tensor_to_voigt(C_tensor):
    """
    Convert a full 4th-order tensor (3x3x3x3) to Voigt notation (6x6).
    """
    C_voigt = np.zeros((6, 6), dtype=np.float32)

    voigt_map = [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1)]

    for I in range(6):
        i, j = voigt_map[I]
        for J in range(6):
            k, l = voigt_map[J]
            C_voigt[I,J] = C_tensor[i,j,k,l]

    return C_voigt


def rotate_stiffness_tensor(C_tensor, R):
    """
    Rotate the stiffness tensor using tensor transformation rules:
    C'_ijkl = R_im * R_jn * R_kp * R_lq * C_mnpq
    """
    C_rotated = np.zeros((3,3,3,3), dtype=np.float32)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    val = 0.0
                    for m in range(3):
                        for n in range(3):
                            for p in range(3):
                                for q in range(3):
                                    val += R[i,m] * R[j,n] * R[k,p] * R[l,q] * C_tensor[m,n,p,q]
                    C_rotated[i,j,k,l] = val

    return C_rotated


def rotate_stiffness_matrix(C, R):
    """
    Rotate the stiffness matrix (Voigt notation).

    Method: Voigt -> Tensor -> Rotate -> Voigt
    This guarantees mathematical correctness.

    Args:
        C: 6x6 Stiffness matrix (Voigt notation)
        R: 3x3 Rotation matrix

    Returns:
        C_rotated: Rotated stiffness matrix (Voigt notation)
    """
    # 1. Voigt -> Tensor
    C_tensor = voigt_to_tensor(C)

    # 2. Rotate tensor
    C_rotated_tensor = rotate_stiffness_tensor(C_tensor, R)

    # 3. Tensor -> Voigt
    C_rotated = tensor_to_voigt(C_rotated_tensor)

    return C_rotated.astype(np.float32)


def stiffness_to_compliance(C):
    """
    Stiffness Matrix C -> Compliance Matrix S
    S = inv(C)
    """
    try:
        S = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        S = np.linalg.pinv(C)

    return S.astype(np.float32)


def compliance_to_engineering(S):
    """
    Compliance Matrix S (6x6) -> Engineering Constants (12 values)

    Returns:
        dict: {'E11', 'E22', 'E33', 'G23', 'G13', 'G12',
               'nu12', 'nu13', 'nu21', 'nu23', 'nu31', 'nu32'}
    """
    # Moduli
    E11 = 1.0 / S[0, 0]
    E22 = 1.0 / S[1, 1]
    E33 = 1.0 / S[2, 2]

    G23 = 1.0 / S[3, 3]
    G13 = 1.0 / S[4, 4]
    G12 = 1.0 / S[5, 5]

    # Poisson's ratios
    nu12 = -S[0, 1] * E11
    nu13 = -S[0, 2] * E11
    nu21 = -S[1, 0] * E22
    nu23 = -S[1, 2] * E22
    nu31 = -S[2, 0] * E33
    nu32 = -S[2, 1] * E33

    return {
        'E11': E11, 'E22': E22, 'E33': E33,
        'G23': G23, 'G13': G13, 'G12': G12,
        'nu12': nu12, 'nu13': nu13,
        'nu21': nu21, 'nu23': nu23,
        'nu31': nu31, 'nu32': nu32
    }


def transform_elastic_constants(label_tensor, R):
    """
    Complete workflow for transforming elastic constants.

    Args:
        label_tensor: shape [12] or [1, 12]
            Order: [E11, E22, E33, G23, G13, G12, nu12, nu13, nu21, nu23, nu31, nu32]
        R: 3x3 Rotation matrix

    Returns:
        transformed_label: Transformed labels
    """
    # Ensure it's 1D
    if label_tensor.dim() == 2:
        label_tensor = label_tensor.squeeze(0)

    # Extract the 12 engineering constants
    E11, E22, E33 = label_tensor[0].item(), label_tensor[1].item(), label_tensor[2].item()
    G23, G13, G12 = label_tensor[3].item(), label_tensor[4].item(), label_tensor[5].item()
    nu12, nu13 = label_tensor[6].item(), label_tensor[7].item()
    nu21, nu23 = label_tensor[8].item(), label_tensor[9].item()
    nu31, nu32 = label_tensor[10].item(), label_tensor[11].item()

    # 1. Engineering constants -> Compliance matrix
    S = engineering_to_compliance(E11, E22, E33, G23, G13, G12,
                                   nu12, nu13, nu21, nu23, nu31, nu32)

    # 2. Compliance matrix -> Stiffness matrix
    C = compliance_to_stiffness(S)

    # 3. Rotate stiffness matrix
    C_rot = rotate_stiffness_matrix(C, R)

    # 4. Rotated stiffness matrix -> Compliance matrix
    S_rot = stiffness_to_compliance(C_rot)

    # 5. Compliance matrix -> Engineering constants
    props_rot = compliance_to_engineering(S_rot)

    # 6. Convert back to tensor
    transformed = torch.tensor([
        props_rot['E11'], props_rot['E22'], props_rot['E33'],
        props_rot['G23'], props_rot['G13'], props_rot['G12'],
        props_rot['nu12'], props_rot['nu13'],
        props_rot['nu21'], props_rot['nu23'],
        props_rot['nu31'], props_rot['nu32']
    ], dtype=label_tensor.dtype)

    return transformed


# ==============================================================================
# 3. Graph Geometric Transformations
# ==============================================================================
def rotate_coordinates(coords, R):
    """
    Rotate spatial coordinates.

    Args:
        coords: shape [N, 3] or [3]
        R: 3x3 Rotation matrix

    Returns:
        rotated_coords: Transformed coordinates
    """
    if isinstance(coords, torch.Tensor):
        R_torch = torch.from_numpy(R).to(coords.dtype).to(coords.device)
        if coords.dim() == 1:
            return coords @ R_torch.T
        else:
            return coords @ R_torch.T
    else:
        if coords.ndim == 1:
            return coords @ R.T
        else:
            return coords @ R.T


def rotate_micro_graph(g_micro, R):
    """
    Rotate geometric features of the micro-scale graph.

    Node Features: [Centroid(3), Verts(12), Vol(1), Props(4)] = 20
    Edge Features: [Direction(3), Dist(1), Normal(3), Area(1), Ratios(4), topo(2)] = 14

    Args:
        g_micro: DGL graph
        R: 3x3 Rotation matrix

    Returns:
        g_rotated: Rotated graph
    """
    g = g_micro.clone()

    # --- Node Features ---
    node_feat = g.ndata['feat']  # [N, 20]
    N = node_feat.shape[0]

    # Extract and rotate centroids (First 3 dims)
    centroids = node_feat[:, :3]  # [N, 3]
    centroids_rot = rotate_coordinates(centroids, R)

    # Extract and rotate vertex coordinates (Dims 4-15, total 12 dims = 4 vertices x 3)
    verts = node_feat[:, 3:15].reshape(N, 4, 3)  # [N, 4, 3]
    verts_rot = rotate_coordinates(verts.reshape(-1, 3), R).reshape(N, 4, 3)
    verts_rot_flat = verts_rot.reshape(N, 12)

    # Remaining features remain unchanged (Vol, Props)
    other_feat = node_feat[:, 15:]  # [N, 5]

    # Reassemble
    g.ndata['feat'] = torch.cat([centroids_rot, verts_rot_flat, other_feat], dim=1)

    # --- Edge Features ---
    if g.num_edges() > 0:
        edge_feat = g.edata['feat']  # [E, 14]

        # Extract and rotate direction vectors (First 3 dims)
        directions = edge_feat[:, :3]  # [E, 3]
        directions_rot = rotate_coordinates(directions, R)
        # Re-normalize to prevent numerical drift
        directions_rot = directions_rot / (torch.norm(directions_rot, dim=1, keepdim=True) + 1e-12)

        # Distance remains unchanged (Dim 4)
        distances = edge_feat[:, 3:4]

        # Extract and rotate normal vectors (Dims 5-7)
        normals = edge_feat[:, 4:7]  # [E, 3]
        normals_rot = rotate_coordinates(normals, R)
        normals_rot = normals_rot / (torch.norm(normals_rot, dim=1, keepdim=True) + 1e-12)

        # Remaining features unchanged (Area, Ratios, is_int)
        other_edge_feat = edge_feat[:, 7:]  # [E, 7]

        # Reassemble
        g.edata['feat'] = torch.cat([directions_rot, distances, normals_rot, other_edge_feat], dim=1)

    return g


def rotate_meso_graph(g_meso, R):
    """
    Rotate geometric features of the meso-scale graph.

    Node Features: [Center(3), Evals(3), Evecs(9), Vol(1), Props(4)] = 20
    Edge Features: [Direction(3), Dist(1), MatProps(4)] = 8

    Args:
        g_meso: DGL graph
        R: 3x3 Rotation matrix

    Returns:
        g_rotated: Rotated graph
    """
    g = g_meso.clone()

    # --- Node Features ---
    node_feat = g.ndata['feat']  # [N, 20]
    N = node_feat.shape[0]

    # Extract and rotate centers (First 3 dims)
    centers = node_feat[:, :3]  # [N, 3]
    centers_rot = rotate_coordinates(centers, R)

    # Eigenvalues unchanged (Dims 4-6)
    evals = node_feat[:, 3:6]

    # Extract and rotate Eigenvectors (Dims 7-15, flattened 3x3 matrix)
    evecs = node_feat[:, 6:15].reshape(N, 3, 3)  # [N, 3, 3]
    # Evecs rotation: R * Evecs
    R_torch = torch.from_numpy(R).to(evecs.dtype).to(evecs.device)
    evecs_rot = torch.matmul(R_torch.unsqueeze(0), evecs)  # [N, 3, 3]
    evecs_rot_flat = evecs_rot.reshape(N, 9)

    # Remaining features unchanged (Vol, Props)
    other_feat = node_feat[:, 15:]  # [N, 5]

    # Reassemble
    g.ndata['feat'] = torch.cat([centers_rot, evals, evecs_rot_flat, other_feat], dim=1)

    # --- Edge Features ---
    if g.num_edges() > 0:
        edge_feat = g.edata['feat']  # [E, 8]

        # Extract and rotate direction vectors (First 3 dims)
        directions = edge_feat[:, :3]  # [E, 3]
        directions_rot = rotate_coordinates(directions, R)
        directions_rot = directions_rot / (torch.norm(directions_rot, dim=1, keepdim=True) + 1e-12)

        # Remaining features unchanged (Dist, MatProps)
        other_edge_feat = edge_feat[:, 3:]  # [E, 5]

        # Reassemble
        g.edata['feat'] = torch.cat([directions_rot, other_edge_feat], dim=1)

    return g


# ==============================================================================
# 4. Complete Data Augmentation Function
# ==============================================================================
def augment_rve_sample(g_micro, g_meso, label, axis='z', angle_deg=90):
    """
    Perform data augmentation on a single RVE sample.

    Args:
        g_micro: Micro-scale graph
        g_meso: Meso-scale graph
        label: Label tensor [12] or [1, 12]
        axis: Rotation axis 'x', 'y', or 'z'
        angle_deg: Rotation angle (degrees)

    Returns:
        g_micro_aug, g_meso_aug, label_aug
    """
    # 1. Obtain rotation matrix
    R = get_rotation_matrix(axis, angle_deg)

    # 2. Rotate graphs
    g_micro_aug = rotate_micro_graph(g_micro, R)
    g_meso_aug = rotate_meso_graph(g_meso, R)

    # 3. Transform elastic constants
    label_aug = transform_elastic_constants(label, R)

    # Maintain original shape
    if label.dim() == 2:
        label_aug = label_aug.unsqueeze(0)

    return g_micro_aug, g_meso_aug, label_aug


# ==============================================================================
# 5. Random Data Augmentation (For Training Loop)
# ==============================================================================
def random_augment(g_micro, g_meso, label, prob=0.5):
    """
    Apply data augmentation randomly.

    Args:
        g_micro, g_meso, label: Original data
        prob: Probability of applying augmentation

    Returns:
        Augmented data (or original data if not triggered)
    """
    import random

    if random.random() > prob:
        return g_micro, g_meso, label

    # Randomly select rotation axis and angle
    axis = random.choice(['x', 'y', 'z'])
    angle = random.choice([90, 180, 270])

    return augment_rve_sample(g_micro, g_meso, label, axis, angle)


# ==============================================================================
# 6. Testing & Validation
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Data Augmentation Module Testing")
    print("=" * 70)

    # Test 1: Rotation Matrix
    print("\n[Test 1] Rotation Matrix")
    R_z90 = get_rotation_matrix('z', 90)
    print(f"90° Rotation around Z-axis:")
    print(R_z90)

    # Test vector rotation: (1, 0, 0) -> (0, 1, 0)
    v = np.array([1, 0, 0])
    v_rot = v @ R_z90.T
    print(f"\nVector Rotation: {v} -> {v_rot}")

    # Test 2: Elastic Constant Transformation
    print("\n[Test 2] Elastic Constant Transformation")

    # Isotropic material (E=200, nu=0.3, G=E/2(1+nu)=76.92)
    label_iso = torch.tensor([
        200.0, 200.0, 200.0,  # E11, E22, E33
        76.92, 76.92, 76.92,  # G23, G13, G12
        0.3, 0.3,             # nu12, nu13
        0.3, 0.3,             # nu21, nu23
        0.3, 0.3              # nu31, nu32
    ])

    print("Original Labels (Isotropic):")
    print(f"  E11={label_iso[0]:.2f}, E22={label_iso[1]:.2f}, E33={label_iso[2]:.2f}")
    print(f"  G12={label_iso[5]:.2f}")

    # Rotate 90° around Z-axis
    label_rot = transform_elastic_constants(label_iso, R_z90)

    print("\nRotated Labels (Should be identical due to isotropy):")
    print(f"  E11={label_rot[0]:.2f}, E22={label_rot[1]:.2f}, E33={label_rot[2]:.2f}")
    print(f"  G12={label_rot[5]:.2f}")

    diff = torch.abs(label_iso - label_rot).max()
    print(f"\nMaximum difference: {diff:.6f} (Should be close to 0)")

    # Test 3: Orthotropic Material
    print("\n[Test 3] Orthotropic Material")
    label_ortho = torch.tensor([
        70000.0, 3000.0, 3000.0,  # E11 >> E22, E33
        1150.0, 1150.0, 1150.0,   # G (Assumed equal for demo)
        0.22, 0.22,               # nu12, nu13
        0.094, 0.3,               # nu21, nu23
        0.094, 0.3                # nu31, nu32
    ])

    print("Original Labels (Fiber direction = X-axis):")
    print(f"  E11={label_ortho[0]:.1f}, E22={label_ortho[1]:.1f}, E33={label_ortho[2]:.1f}")

    # Rotate 90° around Z-axis -> Fiber direction shifts to Y-axis
    label_rot = transform_elastic_constants(label_ortho, R_z90)

    print("\nRotated Labels (Fiber direction = Y-axis):")
    print(f"  E11={label_rot[0]:.1f}, E22={label_rot[1]:.1f}, E33={label_rot[2]:.1f}")
    print("  (E11 and E22 should have swapped values)")

    print("\n✅ Data Augmentation Module Testing Complete")