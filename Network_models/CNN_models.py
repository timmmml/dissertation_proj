"""This module handles the "big step up" networks: CNN-RNN structures"""

import torch
import torch.nn as nn
from .BaseNNAgent import BaseNNAgent
from .RNN_models import FC_RNN, CustomRNN
from utils.path_settings import OBJECT_PATH
import Rotations as Rot
from pytorch3d.transforms import quaternion_to_matrix, quaternion_apply
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesVertex,
    look_at_view_transform,
    FoVPerspectiveCameras
)

def create_conv_layers(model_params):
    conv_layers = []
    for layer_params in model_params.get("conv_layers", []):  # Handle missing key
        layer_type = layer_params.pop("type")

        if layer_type == "Conv2d":
            conv_layers.append(nn.Conv2d(**layer_params))
        elif layer_type == "ReLU":
            conv_layers.append(nn.ReLU())
        elif layer_type == "MaxPool2d":
            conv_layers.append(nn.MaxPool2d(**layer_params))
        # Add more layer types as needed

    return nn.Sequential(*conv_layers)


class Custom_CNN(BaseNNAgent):
    """TODO: write the docstring

    """
    def __init__(self, model_params):
        super(Custom_CNN, self).__init__()
        # Let's set a stereotyped cnn structure for now
        if "conv_layers" in model_params:
            self.conv_layers = create_conv_layers(model_params)
        else:
            self.conv_layers = [
                nn.Conv2d(1, 16, 3, 1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, 3, 1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, 1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            ]
        self.flatten = nn.Flatten()
        self.fc_0 = nn.Linear(2304, model_params.get("fc_dim", 64))
        model_params['input_size'] = model_params.get("fc_dim", 64)
        self.rnn_model = CustomRNN(model_params)
        self.action_period = self.rnn_model.action_period
        self.total_period = self.rnn_model.total_period
        self.stepwise = self.rnn_model.stepwise
        self.out = None
        self.pred = None
        self.dt = self.rnn_model.dt

    def forward(self, x):
        # Assume one channel
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.fc_0(x)
        # Now repeat the convnet output for self.prep_phase times on dim 2, and concatenate with zeros
        x = x.unsqueeze(1).repeat(1, self.prep_phase, 1)
        x = torch.cat((x, torch.zeros(x.size(0), self.total_period - self.prep_phase, x.size(2)).to(x.device)), dim = 1)
        x = self.rnn_model(x)
        self.out = self.rnn_model.out
        self.pred = self.rnn_model.pred
        return x


class Imported_CNN_RNN(BaseNNAgent):
    def __init__(self, model_params):
        super(Imported_CNN_RNN, self).__init__()
        convnet = model_params.get("convnet", "ConvNeXt_Tiny")
        weight_string = model_params.get('weight_string', 'IMAGENET1K_V1')
        rnn = model_params.get('rnn', None)
        self.stepwise = model_params.get('stepwise', False)
        if type(convnet) == str:
            try: #TODO: Heads-up that I am not using the weight transformation here. This may cause problems
                module = __import__("torchvision.models", fromlist=convnet + "_Weights")
                weights = getattr(module, convnet + "_Weights")
                weights = getattr(weights, weight_string)
                module = __import__("torchvision.models", fromlist=convnet.lower())
                model = getattr(module, convnet.lower())
                self.conv = model(weights)
                self.conv = self.conv.features
            except Exception as e:
                raise NotImplementedError(f"{e}, wait for the full version!")
        elif isinstance(convnet, nn.Module):
            self.conv = convnet
        else:
            raise ModuleNotFoundError("No legal convnet specs were provided.")

        if model_params.get("freeze_conv", True):
            self.conv.frozen = True
            for param in self.conv.parameters(recurse = True):
                param.requires_grad = False
        else:
            self.conv.frozen = False
            for param in self.conv.parameters(recurse = True):
                param.requires_grad = True

        self.conv_output_size = self.conv(torch.randn((1, 3, 64, 64))).shape[-3]

        # pool the last layer
        self.postprocess = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.LayerNorm(self.conv_output_size),
        )

        self.postprocess.frozen = False # No params here though

        self.total_period = 100
        self.action_period = 50
        self.prep_period = self.total_period - self.action_period

        self.linear1_size = 64
        self.linear2_size = 32
        self.rnn_input_size = model_params.get("rnn_input_size", 16)

        self.processing = nn.Sequential(
            nn.Linear(self.conv_output_size, self.linear1_size),
            nn.ReLU(),
            nn.Linear(self.linear1_size, self.linear2_size),
            nn.ReLU(),
            nn.Linear(self.linear2_size, self.rnn_input_size),
            nn.ReLU()
        )
        self.processing.frozen = False
        self.output = None
        self.remove_FC = False
        if type(rnn) == dict:
            self.freeze_rnn = rnn.get('freeze_rnn', False)
            self.rnn_hidden_size = rnn.get('hidden_size', 8)
        else:
            self.freeze_rnn = True
        if isinstance(rnn, nn.Module):
            self.rnn = rnn
            self.rnn_hidden_size = self.rnn.hidden_size
            if self.freeze_rnn:
                for param in self.rnn.parameters(recurse = True):
                    param.requires_grad = False
        elif type(rnn) == dict and rnn.get('saved_model_path', None) == None:
            rnn_name = rnn['model_name']
            rnn_module = __import__(f'{rnn["model_path"]}', fromlist = [rnn_name])
            modelClass = getattr(rnn_module, rnn_name)
            self.rnn = modelClass(rnn['model_params'])
            self.rnn_hidden_size = rnn.hidden_size
            print("RNN graph built from configs. Trying to load a state dict")
            self.state_dict_path = rnn.get('saved_state_dict_path', None)
            if self.state_dict_path == None:
                print("No saved params found in the RNN configs. Train the RNN anew.")
                self.rnn.frozen = False
            else:
                try:
                    self.rnn.load_state_dict(torch.load(self.state_dict_path))
                    print("State dict loaded successfully.")
                    for param in self.rnn.parameters(recurse = True):
                        param.requires_grad = False
                    self.rnn.frozen = True
                except Exception as e:
                    print(f"Error loading state dict: {e}")
                    print("Train the RNN anew.")
                    self.rnn.frozen = False
            self.remove_FC = rnn.get('remove_FC', False)

        elif type(rnn) == dict and rnn.get('saved_model_path', None) != None:
            # load directly from path
            self.rnn = torch.load(rnn['saved_model_path'])
            for param in self.rnn.parameters(recurse = True):
                param.requires_grad = False
            self.remove_FC = rnn.get('remove_FC', False)
            self.rnn_hidden_size = self.rnn.hidden_size

        elif type(rnn) == str:
            self.rnn = torch.load(rnn)
            self.rnn_hidden_size = self.rnn.hidden_size
            for param in self.rnn.parameters(recurse = True):
                param.requires_grad = False

        else:
            print("No legal RNN specs were provided. Use a default GRU8. Train anew")
            self.rnn = nn.GRU(self.rnn_input_size, 8, 1, True)  # BATCH FIRST
            self.rnn_hidden_size = 8

        if hasattr(self.rnn, "fc_out"):
            self.output = self.rnn.fc_out
            for param in self.output.parameters(recurse = True):
                param.requires_grad = False
            self.output.frozen = True

        if self.remove_FC:
            self.rnn = self.rnn.rnn
        else:
            raise NotImplementedError("haven't implemented a version where you don't remove FC yet")

        self.rnn.frozen = self.freeze_rnn
        if not self.freeze_rnn:
            for param in self.rnn.parameters(recurse = True):
                param.requires_grad = True
        if self.output is None:
            if hasattr(model_params, "output"):
                if isinstance(model_params['output'], nn.Module):
                    self.output = model_params['output']
                    for param in self.output.parameters(recurse = True):
                        param.requires_grad = False
                    self.output.frozen = True
                elif isinstance(model_params['output'], str):
                    try:
                        output_module = __import__(f"{model_params['output']}", fromlist = [model_params['output']])
                        self.output = output_module
                        for param in self.output.parameters(recurse = True):
                            param.requires_grad = False
                        self.output.frozen = True
                    except Exception as e:
                        print(f"Error loading output module: {e}")
                        print("No output module loaded. Train anew.")
                        self.output = nn.Linear(self.hidden_size, 3)
            else:
                print("No legal output specs were provided. Use a default linear layer. Train anew.")
                self.output = nn.Linear(self.hidden_size, 3)
                self.output.frozen = False

        self.rotated = False


    def forward(self, x):
        """ Implements a simple forward loop.

        Args: x (batch_size, 3, dim1, dim2)
        """
        x = self.conv(x)
        x = self.postprocess(x)
        x = self.processing(x)
        batch_size, processed_size = x.shape
        x = torch.cat((x.unsqueeze(1).repeat(1, self.prep_period, 1), torch.zeros((batch_size, self.action_period,
                                                                                   processed_size), device = x.device)), dim = 1)

        h0 = torch.zeros(1, x.size(0), self.rnn_hidden_size).to(x.device).float()
        x, _ = self.rnn(x, h0)
        self.out = self.output(x).float()
        self.predict()
        return self.out

    def rotate(self, quat, obj_path = None):
        if not hasattr(self, "rotated") or not self.rotated:
            # initialise the rotation
            if obj_path is None:
                # Take the default to be the cow
                obj_path = getattr(self, "object_path", OBJECT_PATH + "\\cow_mesh\\cow.obj")
                print(f"object path {obj_path}")
                obj_path = obj_path if obj_path is not None else OBJECT_PATH + "\\cow_mesh\\cow.obj"
            self.mesh = load_objs_as_meshes([obj_path], device = self.output.bias.device)
            R, T = look_at_view_transform(getattr(self, "cam_position", 2.7), 0, 180)
            print(f"Cam position {getattr(self, 'cam_position', 2.7)}")
            cameras = FoVPerspectiveCameras(device = self.output.bias.device, R = R, T = T)
            raster_settings = RasterizationSettings(
                image_size = getattr(self, "resolution", 256),
                blur_radius = 0.0,
                faces_per_pixel = 1,
            )
            light = PointLights(device = self.output.bias.device, location = [[0.0, 0.0, -3.0]])
            self.renderer = MeshRenderer(
                rasterizer = MeshRasterizer(cameras = cameras, raster_settings = raster_settings),
                shader = SoftPhongShader(device = self.output.bias.device, cameras = cameras, lights = light)
            )
            self.rotated = True
        features = self.renderer(self.mesh.clone().update_padded(quaternion_apply(quat, self.mesh.verts_padded())))[0, ..., :3].permute(2,0,1)
        features = features.unsqueeze(0)
        traj = self(features)[0, -self.action_period:, :].detach()
        traj = Rot.exp_quat(traj * self.dt)
        return traj

class Imported_CNN_Infer(BaseNNAgent):
    def __init__(self, model_params):
        super(Imported_CNN_Infer, self).__init__()
        convnet = model_params.get("convnet", "ConvNeXt_Tiny")
        weight_string = model_params.get('weight_string', 'IMAGENET1K_V1')
        self.stepwise = model_params.get('stepwise', False)
        if type(convnet) == str:
            try: #TODO: Heads-up that I am not using the weight transformation here. This may cause problems
                module = __import__("torchvision.models", fromlist=convnet + "_Weights")
                weights = getattr(module, convnet + "_Weights")
                weights = getattr(weights, weight_string)
                module = __import__("torchvision.models", fromlist=convnet.lower())
                model = getattr(module, convnet.lower())
                self.conv = model(weights)
                self.conv = self.conv.features
            except Exception as e:
                raise NotImplementedError(f"{e}, wait for the full version!")
        elif isinstance(convnet, nn.Module):
            self.conv = convnet
        else:
            raise ModuleNotFoundError("No legal convnet specs were provided.")

        if model_params.get("freeze_conv", True):
            self.conv.frozen = True
            for param in self.conv.parameters(recurse = True):
                param.requires_grad = False
        else:
            self.conv.frozen = False
            for param in self.conv.parameters(recurse = True):
                param.requires_grad = True

        self.conv_output_size = self.conv(torch.randn((1, 3, 64, 64))).shape[-3]

        # pool the last layer
        self.postprocess = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.LayerNorm(self.conv_output_size),
        )

        self.postprocess.frozen = False # No params here though

        self.linear1_size = 64
        self.linear2_size = 32
        self.rnn_input_size = model_params.get("rnn_input_size", 16)

        self.processing = nn.Sequential(
            nn.Linear(self.conv_output_size, self.linear1_size),
            nn.ReLU(),
            nn.Linear(self.linear1_size, self.linear2_size),
            nn.ReLU(),
            nn.Linear(self.linear2_size, self.rnn_input_size),
            nn.ReLU()
        )
        self.processing.frozen = False
        self.output = nn.Linear(self.rnn_input_size, 3)

    def forward(self, x):
        """ Implements a simple forward loop.

        Args: x (batch_size, 3, dim1, dim2)
        """
        x = self.conv(x)
        x = self.postprocess(x)
        x = self.processing(x)
        self.out = self.output(x).float()
        if len(self.out.shape) == 2:
            self.out = self.out.unsqueeze(-2)
        self.predict()
        return self.out








