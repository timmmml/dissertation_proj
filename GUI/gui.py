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
# from pytorch3d.renderer import (
#     look_at_view_transform,
#     FoVPerspectiveCameras,
#     PointLights,
#     DirectionalLights,
#     Materials,
#     RasterizationSettings,
#     MeshRenderer,
#     MeshRasterizer,
#     SoftPhongShader,
#     TexturesUV,
#     TexturesVertex
# )
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader
)
from pytorch3d.transforms import Rotate, RotateAxisAngle, Transform3d, quaternion_apply
import copy
import numpy as np

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the sys.path
sys.path.append(parent_dir)
from utils import goto_project_root
from utils.path_settings import DATA_PATH
import Network_models
import Rotations as Rot

os.environ["IOPATH_LOGLEVEL"] = "DEBUG"
os.environ["IOPATH_DISABLE_TELEMETRY"] = "1"
class MyGLCanvas(glcanvas.GLCanvas):
    def __init__(self, parent, mesh, model_canvas = False, frame = None):
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

        self.renderer = self.init_renderer()

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_TIMER, self.OnTimer)
        self.timer = wx.Timer(self)
        self.timer.Start(50)
        self.frames = []

        self.running = 0
        self.quat = torch.tensor((1, 0, 0, 0), dtype = torch.float32).to(self.device)
        # self.quat sets the current rotation of the mesh

        self.quat_list = []
        self.quat_item = torch.tensor((1, 0, 0, 0)).to(self.device)

        self.time = 0
        self.reset = 0

        self.time_upper = 500 # 500 steps total.
        self.fps = 30

        self.system_default = 1

    def init_renderer(self):
        R, T = look_at_view_transform(2.7, 0, 180)
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
        # self.angle = 0
        # if self.running:
        #     self.angle = 5
        # else:
        #     self.angle = 0
        self.quat = torch.tensor((1, 0, 0, 0), dtype=torch.float32).to(self.device)
        if self.running:
            if not self.model_canvas and self.system_default:  # Left panel rotation, introduction (rotation to the right)
                self.quat = torch.tensor((np.cos(np.pi / 180), 0, np.sin(np.pi / 180), 0)).to(self.device)
            elif not self.model_canvas and not self.system_default:  # Ideal rotation, use the same quaternion for a number of steps
                if self.time == 0:
                    self.parent.log("Ideal rotation started.")
                self.quat = self.quat_item
                self.time += 1
                if self.time == self.time_upper:
                    self.running = 0
                    self.time = 0
                    self.parent.log("Ideal rotation completed.")

            elif self.model_canvas and not self.system_default: # Model rotation, use the list of quaternions
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

    def Render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        # glRotatef(180, 1, 0, 0)

        # R = RotateAxisAngle(self.angle, axis="Y").to(self.device) # Define the current rotation
        # rotated_verts = R.transform_points(self.mesh.verts_padded())
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
        """Different behaviour if or if not this is the model canvas"""
        if not self.model_canvas:
            self.quat_item = self._process_quat(quat)
            self.system_default = 0

        elif self.network is not None:
            try:
                quat = quat.to(self.device)
                self.quat_list =list(self.network.rotate(quat).to(self.device))
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
        """This operation divides the target quaternion into a small quaternion"""
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
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def OnClose(self, event):
        self.Destroy()
        wx.GetApp().ExitMainLoop()

    def OnVisualizeNetwork(self, event):
        self.Hide()
        # Initialize your network, mesh, and angular velocities here
        frame = NetVisFrame(self)
        frame.Show()

    def OnBehavioralExperiment(self, event):
        self.Hide()
        frame = BehavExpFrame(self)
        frame.Show()

    def ShowMasterFrame(self):
        self.Show()



class NetVisFrame(wx.Frame):
    """Handles the visualization of the network, child frame 1."""
    def __init__(self, parent):
        super().__init__(parent, title="OpenGL Mesh Renderer with PyTorch3D", size=(1400, 780))
        self.left_GIF_number = 0
        self.right_GIF_number = 0
        self.left_GIF_name = "Ideal_"
        self.right_GIF_name = "Model_"

        menubar = wx.MenuBar()
        file_menu = wx.Menu()
        load_network = wx.MenuItem(file_menu, wx.ID_OPEN, '&Load Network')
        load_mesh_object = wx.MenuItem(file_menu, wx.ID_OPEN, '&Load Mesh Object')

        file_menu.Append(load_network)
        file_menu.Append(load_mesh_object)

        screenshot_menu = wx.Menu()
        save_gif = wx.MenuItem(screenshot_menu, wx.ID_SAVE, '&Save current comparison as GIF')
        screenshot_menu.Append(save_gif)
        menubar.Append(file_menu, '&File')
        menubar.Append(screenshot_menu, '&Screenshots')

        control_menu = wx.Menu()
        sim_run_button = wx.MenuItem(control_menu, wx.ID_ANY, "&Run Simultaneously")
        reset_button = wx.MenuItem(control_menu, wx.ID_ANY, "&Reset Rotation")
        left_run_button = wx.MenuItem(control_menu, wx.ID_ANY, "&Run Left")
        right_run_button = wx.MenuItem(control_menu, wx.ID_ANY, "&Run Right")
        stop_button = wx.MenuItem(control_menu, wx.ID_ANY, "&Stop All Rotation")
        back_button = wx.MenuItem(control_menu, wx.ID_ANY, "&Back")

        for item in [sim_run_button, reset_button, left_run_button, right_run_button, stop_button,
                     back_button]:
            control_menu.Append(item)

        menubar.Append(control_menu, '&Controls')

        self.SetMenuBar(menubar)

        # Bind events
        self.Bind(wx.EVT_MENU, self.OnLoadNetwork, load_network)
        self.Bind(wx.EVT_MENU, self.OnLoadMeshObject, load_mesh_object)
        self.Bind(wx.EVT_MENU, self.OnSaveGIF, save_gif)
        self.Bind(wx.EVT_MENU, self.OnSimRun, sim_run_button)
        self.Bind(wx.EVT_MENU, self.OnReset, reset_button)
        self.Bind(wx.EVT_MENU, self.OnLeftRun, left_run_button)
        self.Bind(wx.EVT_MENU, self.OnRightRun, right_run_button)
        self.Bind(wx.EVT_MENU, self.OnStop, stop_button)
        self.Bind(wx.EVT_MENU, self.OnBack, back_button)

        # Load two meshes
        self.mesh1 = load_objs_as_meshes(["./data/cow_mesh/cow.obj"])
        self.mesh2 = load_objs_as_meshes(["./data/cow_mesh/cow.obj"])

        self.panel = wx.Panel(self)

        # Create two GL canvases
        self.canvas_left_left = MyGLCanvas(self.panel, self.mesh1, frame = self)
        self.canvas_left_right = MyGLCanvas(self.panel, self.mesh2, 1, frame = self)

        # Create control panel with buttons
        self.control_panel = wx.Panel(self.panel)
        self.init_control_buttons()
        self.init_rotation_inputs()
        self.init_log_box()

        # Arrange GL canvases in a vertical box sizer
        self.gl_sizer = wx.BoxSizer(wx.HORIZONTAL)
        gl_left = wx.BoxSizer(wx.VERTICAL)
        gl_right = wx.BoxSizer(wx.VERTICAL)
        gl_left.Add(wx.StaticText(self.panel, label="Ideal Rotation"), 0, wx.ALL, 5)
        gl_left.Add(self.canvas_left_left, 1, wx.EXPAND | wx.ALL, 5)
        gl_right.Add(wx.StaticText(self.panel, label="Model Rotation"), 0, wx.ALL, 5)
        gl_right.Add(self.canvas_left_right, 1, wx.EXPAND | wx.ALL, 5)
        self.gl_sizer.Add(gl_left, 1, wx.EXPAND | wx.ALL, 5)
        self.gl_sizer.Add(gl_right, 1, wx.EXPAND | wx.ALL, 5)

        # Arrange control panel in a vertical box sizer with fixed width
        self.control_sizer = wx.BoxSizer(wx.VERTICAL)
        self.control_sizer.Add(wx.StaticText(self.control_panel, label="Controls"), 0, wx.ALL, 5)
        self.control_sizer.Add(self.load_network_button, 0, wx.EXPAND | wx.ALL, 5)
        self.control_sizer.Add(self.load_mesh_button, 0, wx.EXPAND | wx.ALL, 5)
        angle_axis_sizer = wx.BoxSizer(wx.HORIZONTAL)
        angle_axis_sizer.Add(self.angle_input, 1, wx.EXPAND | wx.RIGHT, 5)
        angle_axis_sizer.Add(self.axis_input, 1, wx.EXPAND | wx.RIGHT, 5)
        angle_axis_sizer.Add(self.angle_axis_enter_button, 0, wx.EXPAND)

        self.control_sizer.Add(wx.StaticText(self.control_panel,
                                             label="Define rotation (in angle and axis):"), 0, wx.ALL, 5)
        self.control_sizer.Add(angle_axis_sizer, 0, wx.EXPAND | wx.ALL, 5)

        self.control_sizer.Add(wx.StaticText(self.control_panel,
                                             label="Define rotation (in quaternions):"), 0, wx.ALL, 5)
        quat_sizer = wx.BoxSizer(wx.HORIZONTAL)
        quat_sizer.Add(self.quaternion_input, 1, wx.EXPAND | wx.RIGHT, 5)
        quat_sizer.Add(self.quaternion_enter_button, 0, wx.EXPAND)
        self.control_sizer.Add(quat_sizer, 0, wx.EXPAND | wx.ALL, 5)

        self.control_sizer.Add(wx.StaticText(self.control_panel,
                                             label="Define steps to complete the ideal rotation"), 0, wx.ALL, 5)

        steps_sizer = wx.BoxSizer(wx.HORIZONTAL)
        steps_sizer.Add(self.steps_input, 1, wx.EXPAND | wx.RIGHT, 5)
        steps_sizer.Add(self.steps_enter_button, 0, wx.EXPAND)
        self.control_sizer.Add(steps_sizer, 0, wx.EXPAND | wx.ALL, 5)

        self.control_sizer.Add(self.apply_rotation_button, 0, wx.EXPAND | wx.ALL, 5)


        sim_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sim_sizer.Add(self.sim_run_button, 1, wx.EXPAND | wx.ALL, 5)
        sim_sizer.Add(self.stop_both, 1, wx.EXPAND | wx.ALL, 5)
        sim_sizer.Add(self.reset_button, 1, wx.EXPAND | wx.ALL, 5)

        left_sizer = wx.BoxSizer(wx.HORIZONTAL)
        left_sizer.Add(self.left_run_button, 1, wx.EXPAND | wx.ALL, 5)
        left_sizer.Add(self.stop_left, 1, wx.EXPAND | wx.ALL, 5)
        left_sizer.Add(self.reset_left, 1, wx.EXPAND | wx.ALL, 5)

        right_sizer = wx.BoxSizer(wx.HORIZONTAL)
        right_sizer.Add(self.right_run_button, 1, wx.EXPAND | wx.ALL, 5)
        right_sizer.Add(self.stop_right, 1, wx.EXPAND | wx.ALL, 5)
        right_sizer.Add(self.reset_right, 1, wx.EXPAND | wx.ALL, 5)

        self.control_sizer.Add(sim_sizer, 0, wx.EXPAND | wx.ALL, 5)
        self.control_sizer.Add(left_sizer, 0, wx.EXPAND | wx.ALL, 5)
        self.control_sizer.Add(right_sizer, 0, wx.EXPAND | wx.ALL, 5)
        self.control_sizer.Add(self.stop_button, 0, wx.EXPAND | wx.ALL, 5)

        gif_fps_sizer = wx.BoxSizer(wx.HORIZONTAL)
        gif_fps_sizer.Add(wx.StaticText(self.control_panel, label="FPS for GIFs:"), 0, wx.EXPAND | wx.ALL, 5)
        gif_fps_sizer.Add(self.save_gif_fps_input, 1, wx.EXPAND | wx.ALL, 5)
        gif_fps_sizer.Add(self.save_gif_fps_enter, 0, wx.EXPAND | wx.ALL, 5)
        self.control_sizer.Add(gif_fps_sizer, 0, wx.EXPAND | wx.ALL, 5)

        gif_name_sizer = wx.BoxSizer(wx.HORIZONTAL)
        gif_name_sizer.Add(wx.StaticText(self.control_panel, label="Left GIF ID:"), 0, wx.EXPAND | wx.ALL, 5)
        gif_name_sizer.Add(self.left_GIF_name_input, 1, wx.EXPAND | wx.ALL, 5)
        gif_name_sizer.Add(wx.StaticText(self.control_panel, label="Right GIF ID:"), 0, wx.EXPAND | wx.ALL, 5)
        gif_name_sizer.Add(self.right_GIF_name_input, 1, wx.EXPAND | wx.ALL, 5)

        self.control_sizer.Add(gif_name_sizer, 0, wx.EXPAND | wx.ALL, 5)

        gif_sizer = wx.BoxSizer(wx.HORIZONTAL)
        gif_sizer.Add(self.save_gif_left, 1, wx.EXPAND | wx.ALL, 5)
        gif_sizer.Add(self.save_gif_right, 1, wx.EXPAND | wx.ALL, 5)
        self.control_sizer.Add(gif_sizer, 0, wx.EXPAND | wx.ALL, 5)
        self.control_sizer.Add(self.save_gif_button, 0, wx.EXPAND | wx.ALL, 5)
        self.control_sizer.Add(self.back_button, 0, wx.EXPAND | wx.ALL, 5)

        self.control_panel.SetSizer(self.control_sizer)

        self.gl_main = wx.BoxSizer(wx.VERTICAL)
        self.gl_main.Add(self.gl_sizer, 1, wx.EXPAND | wx.ALL, 5)
        self.gl_main.Add(self.log_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        self.main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.main_sizer.Add(self.gl_main, 1, wx.EXPAND | wx.ALL, 5)
        self.main_sizer.Add(self.control_panel, 0, wx.EXPAND | wx.ALL, 5)

        self.panel.SetSizer(self.main_sizer)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

        self.log("Welcome to the Mental Rotations GUI.")
        self.log("Please load a network and a mesh object to begin.")
        self.log("Currently the cow mesh is loaded by default.")
        self.log("Without network loaded, the right canvas will not rotate.")

    def init_control_buttons(self):
        self.load_network_button = wx.Button(self.control_panel, label="Load Network")
        self.load_mesh_button = wx.Button(self.control_panel, label="Load Mesh Object")
        self.save_gif_fps_input = wx.TextCtrl(self.control_panel, value="30")
        self.save_gif_fps_enter = wx.Button(self.control_panel, label="Enter")
        self.left_GIF_name_input = wx.TextCtrl(self.control_panel, value=f"{self.left_GIF_number}")
        self.right_GIF_name_input = wx.TextCtrl(self.control_panel, value=f"{self.right_GIF_number}")
        self.save_gif_left = wx.Button(self.control_panel, label="Save GIF Left")
        self.save_gif_right = wx.Button(self.control_panel, label="Save GIF Right")
        self.save_gif_button = wx.Button(self.control_panel, label="Save GIF Both")
        self.sim_run_button = wx.Button(self.control_panel, label="Run Both")
        self.reset_button = wx.Button(self.control_panel, label="Reset Both")
        self.reset_left = wx.Button(self.control_panel, label="Reset Left")
        self.reset_right = wx.Button(self.control_panel, label="Reset Right")
        self.left_run_button = wx.Button(self.control_panel, label="Run Left")
        self.right_run_button = wx.Button(self.control_panel, label="Run Right")
        self.stop_left = wx.Button(self.control_panel, label="Stop Left")
        self.stop_right = wx.Button(self.control_panel, label="Stop Right")
        self.stop_both = wx.Button(self.control_panel, label="Stop Both")
        self.stop_button = wx.Button(self.control_panel, label="Stop All Rotation")
        self.back_button = wx.Button(self.control_panel, label="Back")

        self.load_network_button.Bind(wx.EVT_BUTTON, self.OnLoadNetwork)
        self.load_mesh_button.Bind(wx.EVT_BUTTON, self.OnLoadMeshObject)
        self.save_gif_fps_input.Bind(wx.EVT_SET_FOCUS, self.OnFocus)
        self.save_gif_fps_enter.Bind(wx.EVT_BUTTON, self.OnSaveGIFFPS)
        self.left_GIF_name_input.Bind(wx.EVT_SET_FOCUS, self.OnFocus)
        self.left_GIF_name_input.Bind(wx.EVT_TEXT, self.OnLeftGIFName)
        self.right_GIF_name_input.Bind(wx.EVT_SET_FOCUS, self.OnFocus)
        self.right_GIF_name_input.Bind(wx.EVT_TEXT, self.OnRightGIFName)
        self.save_gif_left.Bind(wx.EVT_BUTTON, self.OnSaveGIFLeft)
        self.save_gif_right.Bind(wx.EVT_BUTTON, self.OnSaveGIFRight)
        self.save_gif_button.Bind(wx.EVT_BUTTON, self.OnSaveGIF)
        self.sim_run_button.Bind(wx.EVT_BUTTON, self.OnSimRun)
        self.reset_button.Bind(wx.EVT_BUTTON, self.OnReset)
        self.left_run_button.Bind(wx.EVT_BUTTON, self.OnLeftRun)
        self.right_run_button.Bind(wx.EVT_BUTTON, self.OnRightRun)
        self.stop_button.Bind(wx.EVT_BUTTON, self.OnStop)
        self.stop_both.Bind(wx.EVT_BUTTON, self.OnStop)
        self.stop_left.Bind(wx.EVT_BUTTON, self.OnLeftStop)
        self.stop_right.Bind(wx.EVT_BUTTON, self.OnRightStop)
        self.reset_left.Bind(wx.EVT_BUTTON, self.OnResetLeft)
        self.reset_right.Bind(wx.EVT_BUTTON, self.OnResetRight)
        self.back_button.Bind(wx.EVT_BUTTON, self.OnBack)
    def init_rotation_inputs(self):
        self.angle_input = wx.TextCtrl(self.control_panel, value="0")
        self.axis_input = wx.TextCtrl(self.control_panel, value="0, 0, 0")
        self.quaternion_input = wx.TextCtrl(self.control_panel, value="1, 0, 0, 0")
        self.steps_input = wx.TextCtrl(self.control_panel, value="500")

        self.angle_input.Bind(wx.EVT_SET_FOCUS, self.OnFocus)
        self.axis_input.Bind(wx.EVT_SET_FOCUS, self.OnFocus)
        self.quaternion_input.Bind(wx.EVT_SET_FOCUS, self.OnFocus)
        self.steps_input.Bind(wx.EVT_SET_FOCUS, self.OnFocus)

        self.angle_axis_enter_button = wx.Button(self.control_panel, label="Enter")
        self.quaternion_enter_button = wx.Button(self.control_panel, label="Enter")
        self.steps_enter_button = wx.Button(self.control_panel, label="Enter")

        self.angle_axis_enter_button.Bind(wx.EVT_BUTTON, self.OnAngleAxisEnter)
        self.quaternion_enter_button.Bind(wx.EVT_BUTTON, self.OnQuaternionEnter)
        self.steps_enter_button.Bind(wx.EVT_BUTTON, self.OnStepsEnter)

        self.apply_rotation_button = wx.Button(self.control_panel, label="Apply Rotation")
        self.apply_rotation_button.Bind(wx.EVT_BUTTON, self.OnApplyRotation)

        self.target_rotation = np.array([1., 0., 0., 0.])

    def init_log_box(self):
        self.log_box = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL,
                                   size = (900, 150))
        font = wx.Font(10, wx.MODERN, wx.NORMAL, wx.NORMAL)
        self.log_box.SetFont(font)
        sys.stdout = self.Logger(self)

    def OnLeftGIFName(self, event):
        self.left_GIF_number = int(self.left_GIF_name_input.GetValue())

    def OnRightGIFName(self, event):
        self.right_GIF_number = int(self.right_GIF_name_input.GetValue())

    def OnFocus(self, event):
        event.GetEventObject().SetValue("")
        event.Skip()

    def OnSaveGIFFPS(self, event):
        try:
            self.canvas_left_left.fps = int(self.save_gif_fps_input.GetValue())
            self.canvas_left_right.fps = int(self.save_gif_fps_input.GetValue())
            self.log(f"FPS for GIFs set to {self.canvas_left_left.fps}.")
        except ValueError:
            wx.MessageBox("Invalid FPS input", "Error", wx.OK | wx.ICON_ERROR)
            self.log("Invalid FPS input")

    def OnStepsEnter(self, event):
        try:
            steps = int(self.steps_input.GetValue())
            if steps < 1:
                raise ValueError
            self.canvas_left_left.time_upper = steps
            self.log(f"Steps to complete the ideal rotation set to {self.canvas_left_left.time_upper}.")
        except ValueError:
            wx.MessageBox("Invalid steps input", "Error", wx.OK | wx.ICON_ERROR)
            self.log("Invalid steps input")

    def OnAngleAxisEnter(self, event):
        try:
            angle = float(self.angle_input.GetValue())
            axis = np.array([float(x) for x in self.axis_input.GetValue().split(',')])
            if np.linalg.norm(axis) == 0:
                return
            axis = axis / np.linalg.norm(axis)
            qw = np.cos(np.radians(angle) / 2)
            qxyz = axis * np.sin(np.radians(angle) / 2)
            self.target_rotation = torch.tensor([qw, qxyz[0], qxyz[1], qxyz[2]], dtype=torch.float32)
            self.quaternion_input.SetValue(f"{qw:.4f}, {qxyz[0]:.4f}, {qxyz[1]:.4f}, {qxyz[2]:.4f}")
            self.log("Angle-axis rotation defined successfully, press 'Apply Rotation' to apply.")
        except ValueError:
            wx.MessageBox("Invalid angle or axis input", "Error", wx.OK | wx.ICON_ERROR)
            self.log("Invalid angle or axis input")

    def OnQuaternionEnter(self, event):
        try:
            q = torch.tensor([float(x) for x in self.quaternion_input.GetValue().split(',')], dtype=torch.float32)
            if q.norm() == 0:
                return
            q = q / q.norm()
            self.target_rotation = q
            angle = 2 * torch.arccos(q[0])
            axis = q[1:] / torch.sin(angle / 2) if torch.sin(angle / 2) != 0 else torch.tensor([1., 0., 0.])
            self.angle_input.SetValue(f"{np.degrees(angle):.4f}")
            self.axis_input.SetValue(f"{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}")
            self.quaternion_input.SetValue(f"{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}")
            self.log("Quaternion rotation defined successfully, press 'Apply Rotation' to apply.")
        except ValueError:
            wx.MessageBox("Invalid quaternion input", "Error", wx.OK | wx.ICON_ERROR)
            self.log("Invalid quaternion input")

    def OnApplyRotation(self, event):
        # Apply rotation to both canvases
        self.target_rotation = torch.tensor(self.target_rotation, dtype=torch.float32)
        self.target_rotation.to(self.canvas_left_right.device)
        self.canvas_left_left.apply_rotation(self.target_rotation)
        self.canvas_left_right.apply_rotation(self.target_rotation)
        self.log("Rotation applied successfully.")
        self.log(f"Target rotation: {self.target_rotation}, RPY = {Rot.quat_to_euler(self.target_rotation)}")

    def OnLoadNetwork(self, event):
        with wx.FileDialog(self, "Open Network File", wildcard="Model files (*.pth)|*.pth",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

            pathname = fileDialog.GetPath()
            self.network = self.load_network(pathname)
            self.canvas_left_right.network = self.network

            wx.MessageBox('Network Loaded Successfully', 'Info', wx.OK | wx.ICON_INFORMATION)
            self.log("Network loaded successfully, right canvas now set to be controlled by the network.")

    def load_network(self, path):
        network = torch.load(path)
        network.eval()  # Set network to evaluation mode
        return network

    def OnLoadMeshObject(self, event):
        with wx.FileDialog(self, "Open Mesh File", wildcard="OBJ files (*.obj)|*.obj",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

            pathname = fileDialog.GetPath()
            mesh = load_objs_as_meshes([pathname])
            self.canvas_left_left.mesh = mesh
            self.canvas_left_left.mesh_orig = copy.deepcopy(mesh)
            self.canvas_left_right.mesh = mesh
            self.canvas_left_right.mesh_orig = copy.deepcopy(mesh)

            wx.MessageBox('Mesh Object Loaded Successfully', 'Info', wx.OK | wx.ICON_INFORMATION)
            self.log("Mesh object loaded successfully, both canvases now set to display the new mesh object.")

    def OnSaveGIFLeft(self, event):
        filename = Path(DATA_PATH)/Path("saved_images")/Path(self.left_GIF_name + str(self.left_GIF_number) + ".gif")
        self.left_GIF_number += 1
        self.canvas_left_left.SaveGIF(filename)
        self.log(f"{len(self.canvas_left_left.frames)} frames were recorded.")
        self.log(f"Left canvas GIF saved as {filename}")

    def OnSaveGIFRight(self, event):
        filename = Path(DATA_PATH)/Path("saved_images")/Path(self.right_GIF_name + str(self.right_GIF_number) + ".gif")
        self.right_GIF_number += 1
        self.canvas_left_right.SaveGIF(filename)
        self.log(f"Right canvas GIF saved as {filename}")

    def OnSaveGIF(self, event):
        self.left_GIF_number += 1
        self.right_GIF_number += 1
        filename = Path(DATA_PATH)/Path("saved_images")/Path(self.left_GIF_name + str(self.left_GIF_number) + ".gif")
        self.canvas_left_left.SaveGIF(filename)
        filename = Path(DATA_PATH)/Path("saved_images")/Path(self.right_GIF_name + str(self.right_GIF_number) + ".gif")
        self.canvas_left_right.SaveGIF(filename)
        self.log(f"Both canvases GIF saved as {filename}")

    def OnSimRun(self, event):
        # Implement running simulation logic here
        self.canvas_left_left.reset = 0
        self.canvas_left_right.reset = 0
        self.canvas_left_left.running = 1
        self.canvas_left_right.running = 1

    def OnLeftRun(self, event):
        # Implement left run logic here
        self.canvas_left_left.reset = 0
        self.canvas_left_left.running = 1

    def OnRightRun(self, event):
        # Implement right run logic here
        self.canvas_left_right.reset = 0
        self.canvas_left_right.running = 1

    def OnReset(self, event):
        # Implement reset logic here
        self.canvas_left_left.reset = 1
        self.canvas_left_right.reset = 1

    def OnResetLeft(self, event):
        # Implement left reset logic here
        self.canvas_left_left.reset = 1

    def OnResetRight(self, event):
        # Implement right reset logic here
        self.canvas_left_right.reset = 1

    def OnStop(self, event):
        # Implement stop logic here
        self.canvas_left_left.reset = 0
        self.canvas_left_right.reset = 0
        self.canvas_left_left.running = 0
        self.canvas_left_right.running = 0

    def OnLeftStop(self, event):
        # Implement left stop logic here
        self.canvas_left_left.running = 0

    def OnRightStop(self, event):
        # Implement right stop logic here
        self.canvas_left_right.running = 0

    def OnBack(self, event):
        self.Destroy()
        self.GetParent().ShowMasterFrame()

    def log(self, message):
        self.log_box.AppendText(str(message) + "\n")
        self.log_box.ShowPosition(self.log_box.GetLastPosition())

    def OnClose(self, event):
        sys.stdout = sys.__stdout__
        self.Destroy()
        wx.GetApp().ExitMainLoop()

    class Logger:
        """Logger helper for printing to the log box int the GUI."""
        def __init__(self, frame):
            self.frame = frame

        def write(self, message):
            wx.CallAfter(self.frame.log, message)

        def flush(self):
            pass


class BehavExpFrame(wx.Frame):
    """Handles the behavioural experiment, child frame 2."""

    def __init__(self, parent):
        super().__init__(parent, title="Behavioral Experiment", size=(900, 800))
        self.panel = wx.Panel(self)

        # Add GUI elements here for the behavioral experiment
        self.label = wx.StaticText(self.panel, label="Behavioral Experiment Interface", pos=(10, 10))
        self.back_button = wx.Button(self.panel, label="Back")
        self.back_button.Bind(wx.EVT_BUTTON, self.OnBack)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.label, 0, wx.CENTER | wx.ALL, 10)
        sizer.Add(self.back_button, 0, wx.CENTER | wx.ALL, 10)
        self.panel.SetSizer(sizer)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def OnClose(self, event):
        self.Destroy()
        wx.GetApp().ExitMainLoop()

    def OnBack(self, event):
        self.Destroy()
        self.GetParent().ShowMasterFrame()

class MyApp(wx.App):
    def OnInit(self):
        # Initial placeholders for network, mesh, and angular velocities
        # network = None
        # mesh = load_objs_as_meshes(["./data/cow_mesh/cow.obj"], device="cpu")
        # angular_velocities = [torch.tensor([i * 0.1]) for i in range(100)]  # Example angular velocities

        frame = MasterFrame("Mental Rotations")
        sframe = NetVisFrame(frame)
        # frame.Bind(wx.EVT_CLOSE, frame.OnClose)
        sframe.Show()
        return True


if __name__ == "__main__":
    app = MyApp()
    app.MainLoop()
