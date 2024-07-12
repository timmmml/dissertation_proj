import wx
from wx import glcanvas
from OpenGL.GL import *
from OpenGL.GLUT import *
import torch
import pytorch3d
from pytorch3d.io import load_objs_as_meshes
import imageio
from PIL import Image
import sys
import os
from pathlib import Path
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader
)
from pytorch3d.transforms import quaternion_apply
import copy
import numpy as np

class MyGLCanvas(glcanvas.GLCanvas):
    def __init__(self, parent, mesh, model_canvas=False, frame=None):
        attribList = [glcanvas.WX_GL_RGBA, glcanvas.WX_GL_DOUBLEBUFFER, glcanvas.WX_GL_DEPTH_SIZE, 16, 0]
        super().__init__(parent, attribList=attribList)
        self.parent = frame
        self.context = glcanvas.GLContext(self)
        self.mesh = mesh

        self.model_canvas = model_canvas
        self.network = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mesh = self.mesh.to(self.device)
        self.mesh_orig = copy.deepcopy(self.mesh)

        self.camera_position = 2.7  # Initial camera position
        self.renderer = self.init_renderer()

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_TIMER, self.OnTimer)
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        self.timer = wx.Timer(self)
        self.timer.Start(50)
        self.frames = []

        self.running = 0
        self.quat = torch.tensor((1, 0, 0, 0), dtype=torch.float32).to(self.device)

        self.quat_list = []
        self.quat_item = torch.tensor((1, 0, 0, 0)).to(self.device)

        self.time = 0
        self.reset = 0

        self.time_upper = 500
        self.fps = 30

        self.system_default = 1

    def init_renderer(self):
        R, T = look_at_view_transform(self.camera_position, 0, 180)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])

        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights
            )
        )
        return renderer

    def capture_frame(self):
        width, height = self.GetSize()
        buffer = (GLubyte * (3 * width * height))(0)
        glReadBuffer(GL_BACK)
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer)

        image = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 3))

        image = np.flipud(image)

        return image

    def OnSize(self, event):
        self.Refresh()

    def OnPaint(self, event):
        self.SetCurrent(self.context)
        self.Render()

    def OnTimer(self, event):
        self.quat = torch.tensor((1, 0, 0, 0), dtype=torch.float32).to(self.device)
        if self.running:
            if not self.model_canvas and self.system_default:
                self.quat = torch.tensor((np.cos(np.pi / 180), 0, np.sin(np.pi / 180), 0)).to(self.device)
            elif not self.model_canvas and not self.system_default:
                if self.time == 0:
                    self.parent.log("Ideal rotation started.")
                self.quat = self.quat_item
                self.time += 1
                if self.time == self.time_upper:
                    self.running = 0
                    self.time = 0
                    self.parent.log("Ideal rotation completed.")

            elif self.model_canvas and not self.system_default:
                if self.time == 0:
                    self.parent.log("Model rotation started.")
                self.quat = self.quat_list[self.time]
                self.time += 1
                if self.time == self.time_upper:
                    self.running = 0
                    self.time = 0
                    self.parent.log("Model rotation completed.")
        if self.reset:
            self.reset = 0
            self.time = 0
            self.frames = []
            self.quat = torch.tensor((1, 0, 0, 0), dtype=torch.float32).to(self.device)
            self.mesh = copy.deepcopy(self.mesh_orig)
            self.system_default = 1

        self.Refresh()

    def OnKeyDown(self, event):
        keycode = event.GetKeyCode()
        if keycode == ord('W'):
            self.camera_position -= 0.1
            self.update_camera_position()
        elif keycode == ord('S'):
            self.camera_position += 0.1
            self.update_camera_position()

    def update_camera_position(self):
        R, T = look_at_view_transform(self.camera_position, 0, 180)
        self.renderer.rasterizer.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        self.Refresh()

    def Render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        rotated_verts = quaternion_apply(self.quat, self.mesh.verts_padded())
        self.mesh = self.mesh.update_padded(rotated_verts)

        images = self.renderer(self.mesh)
        image = images[0, ..., :3].cpu().numpy()
        image = np.flipud(image)

        glDrawPixels(image.shape[1], image.shape[0], GL_RGB, GL_FLOAT, image)

        self.SwapBuffers()
        if self.running:
            frame = self.capture_frame()
            self.frames.append(frame)

    def SaveGIF(self, filename):
        imageio.mimsave(filename, self.frames, fps=self.fps)

    def apply_rotation(self, quat):
        if not self.model_canvas:
            self.quat_item = self._process_quat(quat)
            self.system_default = 0
        elif self.network is not None:
            try:
                quat = quat.to(self.device)
                self.quat_list = list(self.network.rotate(quat).to(self.device))
                self.time_upper = len(self.quat_list)
            except:
                quat = quat.to(self.device)
                silence_period = self.network.total_period - self.network.action_period
                q_in = quat.unsqueeze(0).unsqueeze(0)
                q_in = q_in.repeat(1, silence_period, 1)
                q_in = torch.cat((q_in, torch.zeros(1, self.network.action_period, 4).to(q_in.device)), dim=1)
                q_in = q_in.to(self.network.fc.bias.device)
                traj = self.network(q_in)
                traj = traj[0, -self.network.action_period, :].detach()
                traj = Rot.exp_quat(traj * self.network.dt)
                self.quat_list = list(traj)
            if not self.network.stepwise:
                self.parent.log(f"predicted rotation: {self.network.pred.to(self.device)}, ideal rotation; {quat}, geodesic distance: {Rot.geodesic_distance(self.network.pred.to(self.device), quat)}")
            else:
                self.parent.log(f"predicted rotation: {self.network.pred[-1, -1, :].to(self.device)}, ideal rotation; {quat}, geodesic distance: {Rot.geodesic_distance(self.network.pred[-1, -1, :].to(self.device), quat)}")
            self.system_default = 0

    def _process_quat(self, quat):
        quat = torch.tensor(quat).to(self.device)
        self.quat_item = Rot.q_slerp(torch.tensor((1, 0, 0, 0), dtype=torch.float32).to(self.device), quat, 1/self.time_upper)
        return self.quat_item

class MasterFrame(wx.Frame):
    def __init__(self, title):
        super().__init__(None, title=title, size=(400, 300))
        self.panel = wx.Panel(self)

        self.network_button = wx.Button(self.panel, label="Visualize Network Activity")
        self.experiment_button = wx.Button(self.panel, label="Behavioral Experiment")

        self.network_button.Bind(wx.EVT_BUTTON, self.OnVisualizeNetwork)
        self.experiment_button.Bind(wx.EVT_BUTTON, self.OnBehavioralExperiment)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.network_button, 0, wx.CENTER | wx.ALL, 10)
        sizer.Add(self.experiment_button, 0, wx.CENTER | wx.ALL, 10)

        self.panel.SetSizer(sizer)

        self.network_frame = None
        self.experiment_frame = None

    def OnVisualizeNetwork(self, event):
        if not self.network_frame:
            self.network_frame = NetworkActivityFrame(title="Network Activity", parent=self)
            self.network_frame.Show()

    def OnBehavioralExperiment(self, event):
        if not self.experiment_frame:
            self.experiment_frame = ExperimentFrame(title="Behavioral Experiment", parent=self)
            self.experiment_frame.Show()

    def log(self, message):
        if self.network_frame:
            self.network_frame.log(message)
        if self.experiment_frame:
            self.experiment_frame.log(message)

class NetworkActivityFrame(wx.Frame):
    def __init__(self, title, parent):
        super().__init__(parent, title=title, size=(800, 600))
        self.parent = parent

        self.splitter = wx.SplitterWindow(self)
        self.canvas_panel = wx.Panel(self.splitter)
        self.log_panel = wx.Panel(self.splitter)

        self.log = wx.TextCtrl(self.log_panel, style=wx.TE_MULTILINE | wx.TE_READONLY)

        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.canvas_sizer = wx.BoxSizer(wx.VERTICAL)
        self.log_sizer = wx.BoxSizer(wx.VERTICAL)

        self.canvas = MyGLCanvas(self.canvas_panel, self.load_mesh(), model_canvas=True, frame=self)

        self.canvas_sizer.Add(self.canvas, 1, wx.EXPAND)
        self.canvas_panel.SetSizer(self.canvas_sizer)

        self.log_sizer.Add(self.log, 1, wx.EXPAND)
        self.log_panel.SetSizer(self.log_sizer)

        self.splitter.SplitHorizontally(self.canvas_panel, self.log_panel)
        self.splitter.SetSashGravity(0.8)

        self.sizer.Add(self.splitter, 1, wx.EXPAND)
        self.SetSizer(self.sizer)

    def log(self, message):
        self.log.AppendText(message + "\n")

    def load_mesh(self):
        obj_filename = os.path.join(os.path.dirname(__file__), "cow_mesh/cow.obj")
        mesh = load_objs_as_meshes([obj_filename], device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        return mesh

    class MyApp(wx.App):
        def OnInit(self):
            self.frame = MasterFrame(title="3D Visualization and Experimentation")
            self.frame.Show()
            return True

    if __name__ == "__main__":
        app = MyApp()
        app.MainLoop()