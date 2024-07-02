from Network_models import Trainer
from utils import goto_project_root
from sklearn.model_selection import KFold
from utils.path_settings import MODEL_SAVE_PATH, DATA_PATH

configs = {
    "model_specs": {
        "model_name": "CustomRNN", # To diffrentiate from the several types of RNN cells.
        "model_path": "Network_models.RNN_models",
        "model_params": {
            "input_size": 4,
            "hidden_size": 64,
            "num_layers": 1,
            "output_size": 3,
            "cell_type": "GRU",
        }
    },

    "device": "cuda",
    "optimizer_specs": {
        "optimizer_name": "Adam",
        "optimizer_params": {
            "lr": 0.001
        }
    },

    "distance_loss": "geodesic",
    "regularisation_loss": "L2",
    "distance_weight": 0.5,

    "save_path": MODEL_SAVE_PATH + "RNN_model",
    "log_path": LOG_PATH,

    "training_config": {
        "task_id": "0.1q",
        "batch_size": 128,
        "seq_len": 1000,
        "prep_phase": 500,
    }
}

configs["training_config"]["save_path"] = DATA_PATH + ("RNN_model_" + configs["training_config"]["task_id"] + ".pth")