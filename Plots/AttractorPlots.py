"""Implement helper functions to plot trajectories on attractors
"""
import numpy as np
import plotly.graph_objects as go
import torch
import os
import imageio
import kaleido


# Animation!
def animate_neural_activity(activity_over_time, x_positions, y_positions, z_positions = None, markersize=10, save_gif=False, gif_name='neural_activity.gif'):
    if save_gif:
        assert gif_name.endswith('.gif'), 'GIF name must end with .gif'
    time_steps = activity_over_time.shape[0]
    num_selected_neurons = activity_over_time.shape[1]
    time_plot = activity_over_time.flatten()
    if z_positions is None:
        z_positions = np.zeros_like(x_positions)
    
    frames = []
    for t in range(time_steps):
        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=x_positions,
                    y=y_positions,
                    z=z_positions,
                    mode='markers',
                    marker=dict(
                        size=markersize,
                        # color=time_plot[t * num_selected_neurons:(t + 1) * num_selected_neurons],
                        # use color to do activity!
                        color = activity_over_time[t].flatten(),
                        colorscale='Viridis',
                        opacity=0.3
                    )
                )
            ],
            name=f'Frame {t}'
        )
        frames.append(frame)

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Z Position',
                aspectratio=dict(x=1, y=1, z=1),
            ),
            title='Neural Activity Animation',
            width=800,
            height=600,
            updatemenus=[{
                'type': 'buttons',
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]
                }]
            }]
        )
    )
    if save_gif:
            # Create a temporary directory to store frames
            os.makedirs("temp_frames", exist_ok=True)
            temp_frame_paths = []

            # Save each frame as a PNG
            for t in range(time_steps):
                fig.update_traces(selector=dict(type="scatter3d"), 
                                marker=dict(color=activity_over_time[t].flatten()))
                temp_frame_path = f"temp_frames/frame_{t:03d}.png"
                fig.write_image(temp_frame_path)
                temp_frame_paths.append(temp_frame_path)

            # Combine PNGs into a GIF
            with imageio.get_writer(gif_name, mode="I", duration=0.1) as writer:
                for temp_frame_path in temp_frame_paths:
                    writer.append_data(imageio.imread(temp_frame_path))

            # Clean up temporary files
            for temp_frame_path in temp_frame_paths:
                os.remove(temp_frame_path)
            os.rmdir("temp_frames")

            print(f"Animation saved as GIF: {gif_name}")

    fig.show()

def animate_neural_activity_jumpy(activity_over_time, x_positions, y_positions, z_positions = None):
    time_steps = activity_over_time.shape[0]
    num_selected_neurons = activity_over_time.shape[1]
    time_plot = activity_over_time.flatten()
    activity_over_time = activity_over_time.numpy()
    if z_positions is None:
        z_positions = np.zeros_like(x_positions)
    
    frames = []
    for t in range(time_steps):
        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=x_positions * (1 + activity_over_time[t]),
                    y=y_positions * (1 + activity_over_time[t]),
                    z=z_positions * (1 + activity_over_time[t]),
                    mode='markers',
                    marker=dict(
                        size=4,
                        # color=time_plot[t * num_selected_neurons:(t + 1) * num_selected_neurons],
                        # use color to do activity!
                        color = activity_over_time[t].flatten(),
                        colorscale='Viridis',
                        opacity=0.8
                    )
                )
            ],
            name=f'Frame {t}'
        )
        frames.append(frame)

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Z Position',
                aspectratio=dict(x=1, y=1, z=1),
            ),
            title='Neural Activity Animation',
            width=800,
            height=600,
            updatemenus=[{
                'type': 'buttons',
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]
                }]
            }]
        )
    )

    fig.show()


def compute_3D_tuning(final, quats, num_angle_bins=5, num_axis_bins=5, percentile=25):
    # percentile=25 means we're looking at the top 75% of neurons
    w = quats[:, 0]
    axis = quats[:, 1:]
    theta = 2 * torch.acos(w)
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-10)

    angle_bins = torch.linspace(0, np.pi, num_angle_bins + 1)
    angle_bin_indices = torch.bucketize(theta, angle_bins) - 1  # Bin indices start from 0

    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    azimuth = torch.atan2(y, x)  # Range: [-π, π]
    elevation = torch.acos(z / torch.norm(axis, dim=1))  # Range: [0, π]

    azimuth_bins = torch.linspace(-np.pi, np.pi, num_axis_bins + 1)
    elevation_bins = torch.linspace(0, np.pi, num_axis_bins + 1)

    azimuth_bin_indices = torch.bucketize(azimuth, azimuth_bins) - 1
    elevation_bin_indices = torch.bucketize(elevation, elevation_bins) - 1
    num_neurons = final.shape[1]
    mean_firing = torch.zeros((num_angle_bins, num_axis_bins, num_axis_bins, num_neurons))

    for i in range(num_angle_bins):
        for j in range(num_axis_bins):
            for k in range(num_axis_bins):
                in_bin = (angle_bin_indices == i) & \
                        (azimuth_bin_indices == j) & \
                        (elevation_bin_indices == k)
                if in_bin.any():
                    mean_firing[i, j, k] = final[in_bin].mean(dim=0)
                else:
                    mean_firing[i, j, k] = torch.nan  # Handle empty bins

    modulation_depths = torch.zeros(num_neurons)

    for n in range(num_neurons):
        neuron_firing = mean_firing[:, :, :, n]
        max_firing = torch.max(neuron_firing.where(~neuron_firing.isnan(), torch.tensor(-np.inf)))
        min_firing = torch.min(neuron_firing.where(~neuron_firing.isnan(), torch.tensor(np.inf)))
        modulation_depths[n] = max_firing - min_firing

    # Define a threshold to select neurons with significant modulation
    threshold = np.nanpercentile(modulation_depths.numpy(), percentile)
    print(f"Threshold: {threshold}")

    selected_neurons = modulation_depths >= threshold
    preferred_angles = torch.zeros(num_neurons)
    preferred_azimuths = torch.zeros(num_neurons)
    preferred_elevations = torch.zeros(num_neurons)

    for n in range(num_neurons):
        if selected_neurons[n]:
            neuron_firing = mean_firing[:, :, :, n].numpy()
            # Find the bin with maximum firing rate
            idx = np.nanargmax(neuron_firing)
            idx_unraveled = np.unravel_index(idx.item(), neuron_firing.shape)
            i_angle, j_azimuth, k_elevation = idx_unraveled
            
            # Compute the preferred rotation parameters
            preferred_angles[n] = angle_bins[i_angle]
            preferred_azimuths[n] = azimuth_bins[j_azimuth]
            preferred_elevations[n] = elevation_bins[k_elevation]
        else:
            preferred_angles[n] = torch.nan
            preferred_azimuths[n] = torch.nan
            preferred_elevations[n] = torch.nan

    # convert these back to my usual way of plotting quaternions: (x, y, z) * theta
    preferred_x, preferred_y, preferred_z = polar2cartesian(preferred_azimuths, preferred_elevations)
    return preferred_angles, torch.stack((preferred_x, preferred_y, preferred_z), dim=-1), selected_neurons
def compute_3D_tuning_revised(final, quats, num_angle_bins=5, num_azimuth_bins=5, num_elevation_bins=5, percentile=25):
    # Ensure quats are normalized
    quats = quats / quats.norm(dim=1, keepdim=True)
    w = quats[:, 0]
    axis = quats[:, 1:]
    theta = 2 * torch.acos(w.clamp(-1.0, 1.0))  # Clamp to handle numerical issues
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-10)
    
    # Angle binning
    angle_bins = torch.linspace(0, np.pi, num_angle_bins + 1)
    angle_bin_indices = torch.bucketize(theta, angle_bins) - 1  # Bin indices start from 0
    
    # Compute azimuth and elevation
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    azimuth = torch.atan2(y, x)  # Range: [-π, π]
    elevation = torch.asin(z.clamp(-1.0, 1.0))  # Range: [-π/2, π/2]
    
    # Azimuth and elevation binning
    azimuth_bins = torch.linspace(-np.pi, np.pi, num_azimuth_bins + 1)
    elevation_bins = torch.linspace(-np.pi/2, np.pi/2, num_elevation_bins + 1)
    
    azimuth_bin_indices = torch.bucketize(azimuth, azimuth_bins) - 1
    elevation_bin_indices = torch.bucketize(elevation, elevation_bins) - 1
    
    num_neurons = final.shape[1]
    mean_firing = torch.full((num_angle_bins, num_azimuth_bins, num_elevation_bins, num_neurons), torch.nan)
    
    # Compute mean firing rates for each bin
    for i in range(num_angle_bins):
        for j in range(num_azimuth_bins):
            for k in range(num_elevation_bins):
                in_bin = (angle_bin_indices == i) & \
                         (azimuth_bin_indices == j) & \
                         (elevation_bin_indices == k)
                if in_bin.any():
                    mean_firing[i, j, k] = final[in_bin].mean(dim=0)
    
    # Calculate modulation depths
    modulation_depths = torch.zeros(num_neurons)
    for n in range(num_neurons):
        neuron_firing = mean_firing[:, :, :, n]
        valid_firing = neuron_firing[~torch.isnan(neuron_firing)]
        if valid_firing.numel() > 0:
            modulation_depths[n] = valid_firing.max() - valid_firing.min()
            max_activity = valid_firing.max()
        
    
    # Select neurons with significant modulation
    threshold = np.nanpercentile(modulation_depths.numpy(), percentile)
    # threshold = np.nanpercentile(max_activity.numpy(), percentile)
    selected_neurons = modulation_depths >= threshold
    
    preferred_angles = torch.full((num_neurons,), torch.nan)
    preferred_azimuths = torch.full((num_neurons,), torch.nan)
    preferred_elevations = torch.full((num_neurons,), torch.nan)
    
    for n in range(num_neurons):
        if selected_neurons[n]:
            neuron_firing = mean_firing[:, :, :, n]
            idx = np.nanargmax(neuron_firing)
            idx_unraveled = np.unravel_index(idx.item(), neuron_firing.shape)
            i_angle, j_azimuth, k_elevation = idx_unraveled
            # Use bin centers as preferred values
            preferred_angles[n] = (angle_bins[i_angle] + angle_bins[i_angle + 1]) / 2
            preferred_azimuths[n] = (azimuth_bins[j_azimuth] + azimuth_bins[j_azimuth + 1]) / 2
            preferred_elevations[n] = (elevation_bins[k_elevation] + elevation_bins[k_elevation + 1]) / 2
    
    # Convert back to Cartesian coordinates
    preferred_x = torch.cos(preferred_elevations) * torch.cos(preferred_azimuths)
    preferred_y = torch.cos(preferred_elevations) * torch.sin(preferred_azimuths)
    preferred_z = torch.sin(preferred_elevations)
    
    preferred_axes = torch.stack((preferred_x, preferred_y, preferred_z), dim=-1)
    # preferred_axes = preferred_axes * (preferred_angles.sign().unsqueeze(-1))
    # preferred_angles = preferred_angles.abs()
 
    return preferred_angles, preferred_axes, selected_neurons


def polar2cartesian(azimuths, elevations):
    x = torch.cos(azimuths) * torch.sin(elevations)
    y = torch.sin(azimuths) * torch.sin(elevations)
    z = torch.cos(elevations)
    return x, y, z

def quat2xs(quat):
    quat = quat * ((quat[:, 0] > 0) * 2 - 1)[:, np.newaxis]
    theta = 2 * np.arccos(quat[:,:1])
    u = quat[:,1:]/(np.linalg.norm(quat[:,1:], axis = 1)[:, np.newaxis])
    xs = theta * u
    return xs

# def quat2rpy(quat, rpy = "xyz"):
#     r = Rot.from_quat(quat, scalar_first=True)
#     return r.as_euler(rpy)

def proj_stereo(q, d = 3): 
    q = q * ((q[:, 0] > 0) * 2 - 1)[:, np.newaxis]
    return q[:,1:]/(d+q[:,:1])

def stereographic_projection(quats):
    # Select a projection point; here, assuming (1, 0, 0, 0)
    # quats = quats / quats.norm(dim=-1, keepdim=True)
    w = quats[:, 0]
    xyz = quats[:, 1:]
    
    # Avoid division by zero by adding a small epsilon to the denominator
    epsilon = 1e-10
    projected_xyz = xyz / (1 - w.view(-1, 1) + epsilon)
    # projected_xyz = projected_xyz / (projected_xyz.norm(dim=-1, keepdim=True))
    
    return projected_xyz

