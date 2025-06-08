"""This module contains functions for implementing rotations.

"""
import numpy as np
from scipy.spatial.transform import Rotation as R
# from scipy.stats import norm
import torch
from numba import jit

def cat_rotations(rotations):
    """Concatenate a list of rotations. Note: this is not to be used for tensors
    This one uses scipy convention. Scalar last.
    """
    r_total = R.from_quat([0, 0, 0, 1])
    for r in rotations:
        r_total = r * r_total
    return r_total


def geodesic_distance(q1, q2):
    """Compute the geodesic distance between two quaternions.

    Args:
        q1(torch.Tensor): A tensor of shape (.,4) representing the quaternion.
        q2(torch.Tensor): A tensor of shape (.,4) representing the quaternion.

    Returns:
        torch.Tensor: The geodesic distance between q1 and q2.
    """
    return 2 * torch.acos(torch.clamp(torch.abs(torch.sum(q1 * q2, dim = -1)), -1, 1))

def integrate_velocities(omega, dt = 1.0):
    """Integrate angular velocities to rotations.

    REAL PART FIRST

    Args:
        omega(torch.Tensor): A tensor of shape (n,3) representing the angular velocity vector.
        dt(float): The time step.

    Returns:
        torch.Tensor: A tensor of shape (.,4) representing the rotation. Scalar first
    """
    assert omega.shape[-1] == 3
    qt = torch.tensor([1.0, 0.0, 0.0, 0.0], device = omega.device) # initial quaternion
    dim = omega.dim()
    i = 3
    while dim > 2:
        qt = qt.unsqueeze(0).expand(omega.shape[-i], -1)
        i += 1
        dim -= 1
    q_list = exp_quat(omega * dt)
    for i in range(omega.shape[-2]):
        q = q_list[..., i, :]
        qt = q_mult(q, qt, scalar_first=True)  # Left-multiply to accumulate rotations
    return qt

def integrate_quat_sequential(q_list):
    """Integrate angular velocities to rotations.

    REAL PART FIRST

    Args:
        q_list(torch.Tensor): A tensor of shape (batch_size, seq_len, 4) representing the quaternion series.

    Returns:
        q_out torch.Tensor: A tensor of shape (batch_size, seq_len, 4) representing the overall rotation at that point in the sequence. Scalar first
    """
    assert q_list.shape[-1] == 4
    qt = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype = torch.float32, device = q_list.device) # initial quaternion
    q_out = torch.zeros_like(q_list)
    for i in range(q_list.shape[0]):
        q = q_list[i, :]
        qt = q_mult(q, qt, scalar_first=True)  # Left-multiply to accumulate rotations
        q_out[i, :] = qt
    q_out = q_out.permute(1, 0, 2)
    return q_out

def exp_quat(omega):
    """Computes the exponential map of an angular velocity vector

    (this should be scaled by time already. It doesn't really matter)
    REAL PART FIRST OUTPUT

    Args:
        omega(torch.Tensor): A tensor of shape (.,3) representing the angular velocity vector.

    Returns:
        torch.Tensor: A quaternion representing the rotation. Scalar first
    """
    theta = torch.norm(omega, dim = -1, keepdim = True)
    theta = theta.clamp(min=1e-8)
    omega_hat = omega / theta
    q = torch.cat([torch.cos(theta / 2), torch.sin(theta / 2) * omega_hat], -1)
    return q

def q_mult(q1, q2, scalar_first = True):
    """Multiply quaternions q1 and q2.

    Args:
        q1(torch.Tensor): A quaternion of shape (.,4).
        q2(torch.Tensor): A quaternion of shape (.,4).

    Returns:
        torch.Tensor: The product of q1 and q2. the returned is scaler first.
    """
    assert q1.shape[-1] == 4 and q2.shape[-1] == 4

    if scalar_first:
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    else:
        x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    M = torch.stack([
        torch.stack([w1, -x1, -y1, -z1], -1),
        torch.stack([x1, w1, -z1, y1], -1),
        torch.stack([y1, z1, w1, -x1], -1),
        torch.stack([z1, -y1, x1, w1], -1)
    ], dim = -2)

    q = torch.stack([w2, x2, y2, z2], -1)

    q_out = torch.matmul(M, q.unsqueeze(-1)).squeeze(-1)

    return q_out

def q_v_mult(q, v, scalar_first=True):
    """
    Rotate a vector v using a quaternion q.

    Args:
        q (torch.Tensor): A quaternion of shape (., 4).
        v (torch.Tensor): A vector of shape (., 3).
        scalar_first (bool): Whether the scalar part of q is the first element.

    Returns:
        torch.Tensor: The rotated vector of shape (., 3).
    """
    assert q.shape[-1] == 4 and v.shape[-1] == 3

    if scalar_first:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    else:
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    v_q = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)  # Shape (., 4)
    q_conj = torch.stack([w, -x, -y, -z], dim=-1) if scalar_first else torch.stack([-x, -y, -z, w], dim=-1)
    v_rotated = q_mult(q_mult(q, v_q, scalar_first), q_conj, scalar_first)

    return v_rotated[..., 1:]  # Shape (., 3)

@jit(nopython=True)
def q_mult_np(q1, q2, scalar_first=True):
    """Multiplication of quaternions, this time in numpy

    Args:
        q1(np.array): A quaternion of shape (4,).
        q2(np.array): A quaternion of shape (4,).

    Returns:
        np.array: The product of q1 and q2.
    """

    assert q1.shape[-1] == 4 and q2.shape[-1] == 4

    if scalar_first:
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    else:
        x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
        x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]

    M = np.array([
        [w1, -x1, -y1, -z1],
        [x1, w1, -z1, y1],
        [y1, z1, w1, -x1],
        [z1, -y1, x1, w1]
    ])
    q = np.array([w2, x2, y2, z2])

    q_out = M@q

    return q_out


def q_conjugate(q):
    """Compute the conjugate of a quaternion.
    REAL PART FIRST

    Args:
        q(torch.Tensor): A quaternion of shape (.,4).

    Returns:
        torch.Tensor: The conjugate of q.
    """
    assert q.shape[-1] == 4
    return torch.cat([q[..., 0:1], -q[..., 1:4]], -1)

def q_slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions.

    Args:
        q1 (torch.Tensor): A tensor of shape (.,4) representing the quaternion.
        q2 (torch.Tensor): A tensor of shape (.,4) representing the quaternion.
        t (float): The interpolation parameter (0 to 1)

    Returns:
        torch.Tensor: The interpolated quaternion.
    """
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
    q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)

    # Compute the cosine of the angle between the two vectors
    dot = torch.sum(q1 * q2, dim=-1, keepdim=True)

    # If the dot product is negative, reverse one quaternion
    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.abs(dot)

    # Compute the angle theta between q1 and q2
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)

    # Compute interpolation angles
    theta = theta_0 * t
    sin_theta = torch.sin(theta)

    # Compute the coefficients
    s1 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0

    # Perform the interpolation
    q_s = s1 * q1 + s2 * q2
    q_s = q_s / torch.norm(q_s, dim=-1, keepdim=True)
    return q_s

def quat_to_euler(q):
    """Convert a quaternion to Euler angles.

    Args:
        q(torch.Tensor): A tensor of shape (.,4) representing the quaternion.

    Returns:
        torch.Tensor: A tensor of shape (.,3) representing the Euler angles.
    """
    assert q.shape[-1] == 4
    return R.from_quat(q.cpu().numpy()).as_euler('xyz')

def quat_to_mat(q): 
    """Convert a quaternion to a rotation matrix.

    Args:
        q(torch.Tensor): A tensor of shape (.,4) representing the quaternion.

    Returns:
        torch.Tensor: A tensor of shape (.,3,3) representing the rotation matrix.
    """
    assert q.shape[-1] == 4
    return torch.tensor(R.from_quat(q.cpu().numpy(), scalar_first=True).as_matrix()).to(q.device)