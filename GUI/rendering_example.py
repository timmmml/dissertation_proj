import wx
from wx import glcanvas
import numpy as np
import torch
from OpenGL.GL import *
from OpenGL.GLUT import *
from pytorch3d.io import load_objs_as_meshes
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

class MyGLCanvas(glcanvas.GLCanvas):
    def __init__(self, parent, mesh):
        attribList = [glcanvas.WX_GL_RGBA, glcanvas.WX_GL_DOUBLEBUFFER, glcanvas.WX_GL_DEPTH_SIZE, 16, 0]
        super().__init__(parent, attribList=attribList)
        self.context = glcanvas.GLContext(self)
        self.mesh = mesh

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mesh = self.mesh.to(self.device)
        self.mesh_orig = copy.deepcopy(self.mesh)

        self.renderer = self.init_renderer()

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_TIMER, self.OnTimer)
        self.timer = wx.Timer(self)
        self.timer.Start(50)

        self.running = 0
        self.quat = torch.tensor((1, 0, 0, 0)).to(self.device)
        self.reset = 0

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
        self.quat = torch.tensor((1, 0, 0, 0)).to(self.device)
        if self.running:
            self.quat = torch.tensor((np.cos(np.pi / 180), 0, np.sin(np.pi / 180), 0)).to(self.device)

        if self.reset:
            self.reset = 0
            self.mesh = copy.deepcopy(self.mesh_orig)
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


class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="OpenGL Mesh Renderer with PyTorch3D", size=(1200, 560))

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

        for item in [sim_run_button, reset_button, left_run_button, right_run_button, stop_button, back_button]:
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
        self.mesh1 = load_objs_as_meshes(["../data/cow_mesh/cow.obj"])
        self.mesh2 = load_objs_as_meshes(["../data/cow_mesh/cow.obj"])

        self.panel = wx.Panel(self)

        # Create two GL canvases
        self.canvas_left_left = MyGLCanvas(self.panel, self.mesh1)
        self.canvas_left_right = MyGLCanvas(self.panel, self.mesh2)

        # Create control panel with buttons
        self.control_panel = wx.Panel(self.panel)
        self.init_control_buttons()

        # Arrange GL canvases in a vertical box sizer
        self.gl_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.gl_sizer.Add(self.canvas_left_left, 1, wx.EXPAND | wx.ALL, 5)
        self.gl_sizer.Add(self.canvas_left_right, 1, wx.EXPAND | wx.ALL, 5)

        # Arrange control panel in a vertical box sizer with fixed width
        self.control_sizer = wx.BoxSizer(wx.VERTICAL)
        self.control_sizer.Add(self.load_network_button, 0, wx.EXPAND | wx.ALL, 5)
        self.control_sizer.Add(self.load_mesh_button, 0, wx.EXPAND | wx.ALL, 5)
        self.control_sizer.Add(self.save_gif_button, 0, wx.EXPAND | wx.ALL, 5)
        self.control_sizer.Add(self.sim_run_button, 0, wx.EXPAND | wx.ALL, 5)
        self.control_sizer.Add(self.reset_button, 0, wx.EXPAND | wx.ALL, 5)
        self.control_sizer.Add(self.left_run_button, 0, wx.EXPAND | wx.ALL, 5)
        self.control_sizer.Add(self.right_run_button, 0, wx.EXPAND | wx.ALL, 5)
        self.control_sizer.Add(self.stop_button, 0, wx.EXPAND | wx.ALL, 5)
        self.control_sizer.Add(self.back_button, 0, wx.EXPAND | wx.ALL, 5)

        self.control_panel.SetSizer(self.control_sizer)

        # Arrange the main sizer to contain the GL sizer and the control panel
        self.main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.main_sizer.Add(self.gl_sizer, 1, wx.EXPAND | wx.ALL, 5)
        self.main_sizer.Add(self.control_panel, 0, wx.EXPAND | wx.ALL, 5)

        self.panel.SetSizer(self.main_sizer)

    def init_control_buttons(self):
        self.load_network_button = wx.Button(self.control_panel, label="Load Network")
        self.load_mesh_button = wx.Button(self.control_panel, label="Load Mesh Object")
        self.save_gif_button = wx.Button(self.control_panel, label="Save GIF")
        self.sim_run_button = wx.Button(self.control_panel, label="Run Simultaneously")
        self.reset_button = wx.Button(self.control_panel, label="Reset Rotation")
        self.left_run_button = wx.Button(self.control_panel, label="Run Left")
        self.right_run_button = wx.Button(self.control_panel, label="Run Right")
        self.stop_button = wx.Button(self.control_panel, label="Stop All Rotation")
        self.back_button = wx.Button(self.control_panel, label="Back")

        self.load_network_button.Bind(wx.EVT_BUTTON, self.OnLoadNetwork)
        self.load_mesh_button.Bind(wx.EVT_BUTTON, self.OnLoadMeshObject)
        self.save_gif_button.Bind(wx.EVT_BUTTON, self.OnSaveGIF)
        self.sim_run_button.Bind(wx.EVT_BUTTON, self.OnSimRun)
        self.reset_button.Bind(wx.EVT_BUTTON, self.OnReset)
        self.left_run_button.Bind(wx.EVT_BUTTON, self.OnLeftRun)
        self.right_run_button.Bind(wx.EVT_BUTTON, self.OnRightRun)
        self.stop_button.Bind(wx.EVT_BUTTON, self.OnStop)
        self.back_button.Bind(wx.EVT_BUTTON, self.OnBack)

    def OnLoadNetwork(self, event):
        # Implement loading network logic here
        pass

    def OnLoadMeshObject(self, event):
        # Implement loading mesh object logic here
        pass

    def OnSaveGIF(self, event):
        # Implement saving GIF logic here
        pass

    def OnSimRun(self, event):
        # Implement running simulation logic here
        self.canvas_left_left.reset = 0
        self.canvas_left_right.reset = 0
        self.canvas_left_left.running = 1
        self.canvas_left_right.running = 1

    def OnReset(self, event):
        # Implement reset logic here
        self.canvas_left_left.reset = 1
        self.canvas_left_right.reset = 1

    def OnLeftRun(self, event):
        # Implement left run logic here
        pass

    def OnRightRun(self, event):
        # Implement right run logic here
        pass

    def OnStop(self, event):
        # Implement stop logic here
        self.canvas_left_left.reset = 0
        self.canvas_left_right.reset = 0
        self.canvas_left_left.running = 0
        self.canvas_left_right.running = 0

    def OnBack(self, event):
        # Implement back logic here
        pass


class MyApp(wx.App):
    def OnInit(self):
        frame = MainFrame()
        frame.Show()
        return True


if __name__ == "__main__":
    app = MyApp()
    app.MainLoop()
