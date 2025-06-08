import torch
import numpy as np
from Rotations import *

class Manifold:
    def integrate(self, velocities, dt):
        """Integrate velocities to obtain positions on the manifold."""
        raise NotImplementedError

    def distance(self, x, y):
        """Compute distances between positions x and y on the manifold."""
        raise NotImplementedError


class ModifiedRingManifold(Manifold):
    def __init__(self, R=1.0, A=0.2, n=3):
        """
        R: Base radius of the ring.
        A: Amplitude of the sine modulation.
        n: Frequency of the sine modulation.
        """
        self.R = R
        self.A = A
        self.n = n

    def integrate_2(self, angular_velocities, dt):
        """
        Integrate angular velocities to positions on the modified ring manifold.
        
        angular_velocities: Tensor of shape (N, T, 1)
        dt: Time step
        """
        # Integrate angular velocities to get theta over time
        theta = torch.sum(angular_velocities * dt, dim=1)  # Shape: (N, T, 1)

        # Modulate radius with sine function
        r = self.R + self.A * torch.sin(self.n * theta)

        # Convert to Cartesian coordinates
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        positions = torch.cat([x, y], dim=-1)  # Shape: (N, T, 2)
        return positions  # Return positions over time

    def integrate(self, tangent_velocities, dt):
        """
        Integrate velocities in the manifold's tangent space to obtain positions on the manifold.
        
        tangent_velocities: Tensor of shape (N, T, 1)
            The velocities along the manifold at each time step.
        dt: Time step size (scalar)
        
        Returns:
        positions: Tensor of shape (N, T, 2)
            The positions (x, y) on the manifold over time.
        """
        N, T, _ = tangent_velocities.shape
        
        # Initialize theta, x, y
        theta = torch.zeros((N, T), device=tangent_velocities.device)
        x = torch.zeros((N, T), device=tangent_velocities.device)
        y = torch.zeros((N, T), device=tangent_velocities.device)
        
        # Set initial theta (can be random or zero)
        theta[:, 0] = 0.0  # Initial theta; you can randomize this if needed
        
        for t in range(T - 1):
            theta_t = theta[:, t]  # Shape: (N,)
            v_t = tangent_velocities[:, t, 0]  # Shape: (N,)
            
            # Compute norm of the tangent vector ||T(theta_t)||
            sin_n_theta = torch.sin(self.n * theta_t)
            cos_n_theta = torch.cos(self.n * theta_t)
            R_plus_Asin = self.R + self.A * sin_n_theta
            T_norm = torch.sqrt(
                (R_plus_Asin)**2 + (self.A * self.n * cos_n_theta)**2
            )  # Shape: (N,)
            
            # Update theta
            dtheta_dt = v_t / T_norm  # Shape: (N,)
            theta[:, t + 1] = theta[:, t] + dtheta_dt * dt  # Shape: (N,)
            
            # Optionally wrap theta within [0, 2π]
            theta[:, t + 1] = theta[:, t + 1] % (2 * np.pi)
            
            # Compute x and y at time t
            R_plus_Asin_t = self.R + self.A * sin_n_theta  # Shape: (N,)
            x[:, t] = R_plus_Asin_t * torch.cos(theta_t)
            y[:, t] = R_plus_Asin_t * torch.sin(theta_t)
        
        # Compute x and y at final time step T-1
        theta_T = theta[:, -1]
        sin_n_theta_T = torch.sin(self.n * theta_T)
        R_plus_Asin_T = self.R + self.A * sin_n_theta_T
        x[:, -1] = R_plus_Asin_T * torch.cos(theta_T)
        y[:, -1] = R_plus_Asin_T * torch.sin(theta_T)
        
        positions = theta_T.unsqueeze(-1) 
        # # Stack x and y to get positions of shape (N, T, 2)
        positions = torch.stack([x, y], dim=-1)
        positions = positions[:, -1] # Return only the final positions
        
        return positions

    def distance(self, x, y):
        """
        Compute approximate geodesic distance between positions x and y on the manifold.
        
        x, y: Tensors of shape (..., 2)
        """
        # Compute Euclidean distance as an approximation
        # For more accurate results, compute the path length along the manifold
        print(x.shape, y.shape)
        delta = ((x.unsqueeze(1) - y.unsqueeze(0)) + np.pi) % (2 * np.pi) - np.pi
        dist = torch.norm(delta, dim=-1)  # Euclidean distance
        # dist = torch.abs(delta)
        return dist

class Torus(Manifold): 
    def __init__(self, dim=1, R=torch.tensor([1.0])):
        """
        dim (int): Dimension of the torus 
        R (tensor(dim,)): Radii of the torus
        """
        assert dim > 0, "Dimension must be greater than 0."
        assert isinstance(R, torch.Tensor), "R must be a tensor."
        assert len(R.shape) == 1, "R must be a 1D tensor."
        if len(R) != dim:
            if len(R) == 1:
                R = torch.repeat_interleave(R, dim)
            else: 
                raise ValueError("Length of R must be 1 or equal to dim.")
        self.R = R
        self.dim = dim
    
    def integrate(self, angular_velocities, dt):
        theta = torch.sum(angular_velocities * dt, dim=-2)  # Shape: (Batch_num, dim)
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        xyz = theta * self.R.unsqueeze(0).to(angular_velocities.device) # Shape: (Batch_num, dim)
        return xyz
    
    def distance(self, x, y):
        return torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=-1)

class Sphere(Manifold):
    def __init__(self, dim=1):
        """
        dim (int): Dimension of the sphere
        R (tensor(dim,)): Radii of the sphere
        """
        assert dim > 0, "Dimension must be greater than 0."
        assert dim < 4, "I don't know how to handle dimensions greater than 3."

        self.dim = dim
    
    def integrate(self, angular_velocities, dt):
        assert angular_velocities.shape[-1] == self.dim, "Dimension of angular velocities must be equal to dim."
        if self.dim < 3: 
            # expand to 3D
            angular_velocities = torch.cat([angular_velocities, torch.zeros(angular_velocities.shape[:-1] + (3 - self.dim,), device=angular_velocities.device)], dim=-1)

        # treat as rotations
        q = integrate_velocities(angular_velocities * dt)
        q = q * torch.sign(q[..., 0:1])  # Ensure positive scalar component

        if self.dim == 1: 
            # convert to 1D
            q = q[..., 0:2]
        
        if self.dim == 2: 
            # ROTATE A VECTOR BY THE QUATERNION
            unit_vector = torch.tensor([0.0, 0.0, 1.0], device=q.device)
            rotated_vector = q_v_mult(q, unit_vector)
            return rotated_vector

        return q


    def distance(self, xs, ys): 
        """Compute the distance matrix between two sets of points on the sphere."""
        if self.dim==2:
            return (xs.unsqueeze(1) - ys.unsqueeze(0)).norm(dim=-1)
        return geodesic_distance_matrix(xs, ys)

    

def geodesic_distance(u, v, radius=1.0):
    """
    Compute the geodesic distance between two unit vectors on the sphere^(N-1)
    
    u, v (Tensor(..., N)): representing unit vectors.
    radius (float): Radius of the sphere (default is 1.0).
    
    Returns:
    distance (Tensor(...)): Geodesic distance between u and v.
    """
    gamma = torch.acos(torch.clamp(torch.sum(u * v, dim=-1), -1, 1))
    distance = radius * gamma
    return distance

def geodesic_distance_elliptic(u, v, radii):
    """
    Compute the geodesic distance between two points on an elliptic sphere.
    
    Args:
        u (torch.Tensor): A tensor of shape (..., N) representing unit vectors on the elliptic sphere.
        v (torch.Tensor): A tensor of shape (..., N) representing unit vectors on the elliptic sphere.
        radii (torch.Tensor or list): A 1D tensor or list of length N representing the semi-axes of the ellipsoid.
    
    Returns:
        torch.Tensor: A tensor of shape (...) representing the geodesic distance between u and v.
    """
    assert u.shape[-1] == v.shape[-1] == len(radii), "Dimension mismatch between inputs and radii."
    
    radii_inv = torch.tensor(radii, device=u.device).reciprocal()
    u_rescaled = u * radii_inv
    v_rescaled = v * radii_inv
    
    dot_product = torch.sum(u_rescaled * v_rescaled, dim=-1)
    gamma = torch.acos(torch.clamp(dot_product, -1, 1))
    
    radius_effective = torch.sqrt(torch.mean(torch.tensor(radii) ** 2))
    
    distance = radius_effective * gamma
    return distance

def geodesic_distance_matrix(xs, ys):
    """
    Compute the geodesic distance matrix between two batches of points on an elliptic sphere.

    Args:
        xs (torch.Tensor): A tensor of shape (batch_size, len_xs, N), where N is the dimensionality of the vectors.
        ys (torch.Tensor): A tensor of shape (batch_size, len_ys, N), where N is the dimensionality of the vectors.
        radii (torch.Tensor or list): A 1D tensor or list of length N representing the semi-axes of the ellipsoid.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, len_xs, len_ys) representing the geodesic distance matrix.

    """
    # assert xs.shape[-1] == ys.shape[-1] == len(radii), "Dimension mismatch between inputs and radii."

    # dot_products = torch.einsum('bik,bjk->bij',xs, ys)
    gamma = torch.acos(torch.clamp(torch.abs(xs @ ys.T), -1, 1))

    # gamma = torch.acos(torch.clamp(dot_products, -1, 1))
    return gamma