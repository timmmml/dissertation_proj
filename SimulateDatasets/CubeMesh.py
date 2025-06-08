""" This is my interface for the cubemesh object.

Attributes: 
    camera_position: the position of the camera, fixed (except when manually set to be different)
    vertices: 8 x 3 tensor of the matrices of the cube
    faces: 12 x 3 tensor of the faces of the cube
    raster_settings: the settings for the rasterizer (set the preferred image size here)
    lights: the lights in the scene
    current_mode: ...
    processed_quat: ...

Methods:
    quat2verts:
        Args: 
            q (..., 4): tensor of quaternions, optional (default: self.processed_quat; if that's None throw an error)
            occlusion (str): the occlusion mode specifying what information to provide in the third dimension (default: "camera_simple" (_camera coordinates_, 2D), options: "binary" (_projected coordinates_ x visible or not), "full" (actual coordinates, 3D), "camera_simple" (camera coordinates, 2D)), "camera" (camera coordinates, 2D x camera position) ("camera" is not implemented yet)
        Returns:
            verts (..., 8, 3 or 2): tensor of the vertices of the cube (the last dimension depends on the occlusion mode)
            sets current_mode, processed_quat to the input values

    render: 
        Args: 
            q (..., 4): optional (default: self.processed_quat; if that's return a warning with unrotated cube)
        
        Returns:
            image (..., 1, 16, 16): default to a very low resolution, but can be changed with the raster_settings

"""

import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesVertex,
)
from pytorch3d.transforms import quaternion_apply, quaternion_invert


class CubeMesh:
    def __init__(self, camera_position=torch.tensor([0.0, 0.0, 3.0]), resolution=16, default_mode="camera_simple"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.camera_position = camera_position.to(self.device)

        self.vertices = torch.tensor(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ],
            dtype=torch.get_default_dtype(),
        ).to(self.device)

        self.faces = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
                [4, 5, 6],
                [5, 7, 6],
                [0, 1, 4],
                [1, 5, 4],
                [2, 3, 6],
                [3, 7, 6],
                [0, 2, 4],
                [2, 6, 4],
                [1, 3, 5],
                [3, 7, 5],
            ],
            dtype=torch.get_default_dtype(),
        ).to(self.device)
        white_color = torch.ones_like(self.vertices).unsqueeze(0)
        textures = TexturesVertex(verts_features=white_color)

        self.mesh = Meshes(verts=[self.vertices], faces=[self.faces], textures=textures)

        self.raster_settings = RasterizationSettings(
            image_size=resolution, blur_radius=0.0, faces_per_pixel=1
        )
        self.lights = PointLights(device=self.device, location= [[0, 0, 5.2]])

        self.cameras = FoVPerspectiveCameras(
            device=self.device,
            R=torch.eye(3).unsqueeze(0),
            T=self.camera_position.unsqueeze(0),
        )

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings),
            shader=SoftPhongShader(device=self.device, cameras=self.cameras, lights=self.lights),
        )
    
        self.processed_quat = None
        self.default_mode = default_mode

    def quat2verts(self, q = None, occlusion=None):
        if q is None:
            if self.processed_quat is None:
                raise ValueError("No quaternion provided")
            q = self.processed_quat
        
        if occlusion is None:
            occlusion = self.default_mode

        q = q/q.norm(dim=-1, keepdim=True)
        self.processed_quat = q
        q_shape = q.shape
        q = q.to(self.device).view(-1, 4)
        verts = self.vertices.repeat(q.shape[:-1] + (1, 1))
        q = q.unsqueeze(-2).expand(q.shape[:-1] + (8, 4))

        rotated_vertices = quaternion_apply(q, verts)

        if occlusion == "camera_simple":
            transformed_vertices = self.cameras.get_full_projection_transform().transform_points(rotated_vertices)
            projected_vertices = transformed_vertices[..., :2]/transformed_vertices[..., 2].unsqueeze(-1)
            projected_vertices = projected_vertices.view(*q_shape[:-1], 16)
            return projected_vertices
        else: 
            projected_vertices = rotated_vertices[..., :2]
        if occlusion == "full": 
            occ = (rotated_vertices[..., 2]).float().unsqueeze(-1)
        elif occlusion == "binary":
            occ = (rotated_vertices[..., 2] > 0).float().unsqueeze(-1)
        else: 
            raise ValueError("Invalid occlusion mode")

        output = torch.cat([projected_vertices, occ], dim=-1)
        output = output.view(*q_shape[:-1], 24)
        return output

    def render_cube(self, q = None, cube = None):
        """
        Renders the cube with lighting, camera, and rasterization settings.
        """
        if cube is None:
            if q is None:
                if self.processed_quat is None:
                    raise ValueError("No quaternion provided")
                q = self.processed_quat

            q = q.to(self.device)

            q_shape = q.shape
            q = q.view(-1, 4)
            q = q.unsqueeze(-2).expand(-1, 8, 4)
            # my guess is that you don't need to do minibatch here - these are small meshes.

            meshes = self.mesh.extend(len(q))
            rotated_meshes = meshes.update_padded(
                quaternion_apply(q.to(self.device), meshes.verts_padded())
            )
            # print(f"memory after mesh rotation: {torch.cuda.memory_allocated()/1000} KiB")
            imag = self.renderer(rotated_meshes)[..., :1].permute(0, 3, 1, 2)
            imag = imag.view(*q_shape[:-1], 1, *imag.shape[-2:])
        else: # if cube is provided, render whatever is in there. 
            # Cube contains something of (..., 8, 3) shape, we interpret that as vertices
            mesh = Meshes(verts=[cube], faces=[self.faces], textures=self.mesh.textures)
            imag = self.renderer(mesh)[..., :1].permute(0, 3, 1, 2)
    
        return imag

    def to(self, device):
        self.device = device
        self.vertices = self.vertices.to(device)
        self.faces = self.faces.to(device)
        self.mesh = self.mesh.to(device)
        self.cameras = self.cameras.to(device)
        self.renderer = self.renderer.to(device)
        return self

