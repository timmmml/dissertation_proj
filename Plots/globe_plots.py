import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from importlib import reload
import mgplvm as mgp
import SimulateDatasets.GenTrainingData as g
import Rotations as Rot

def get_hidden_state(module, input, output):
    global hidden_states
    hidden_states.append(output[0].detach().clone())


def plot_tuning_curves_GT(trainer, configs, n_size, activation_path, gen_test_data=True, show=False, noise=False):
    global hidden_states
    device = mgp.utils.get_device()
    n_ts = 1600

    if not os.path.isfile(configs['training_config']['data_save_path']) or gen_test_data:
        # generate quaternions over time using an autoregressive process
        qs_t = np.random.normal(0, 1, (n_ts, 4))  # random points on sphere
        qs_t = qs_t / np.sqrt(np.sum(qs_t ** 2, axis=1, keepdims=True))  # normalize
        for t in range(1, n_ts):  # autoregressive process
            step = np.random.normal(0, 0.4, 3)  # displacement in tangent space
            theta = np.sqrt(np.sum(step ** 2))
            v = step / theta
            qt = np.concatenate(
                [np.ones(1) * np.cos(theta), np.sin(theta) * v])  # displacement in quaternion space
            qs_t[t, :] = Rot.q_mult_np(qt, qs_t[t - 1, :])

        qs_t = np.sign(qs_t[:, :1]) * qs_t  # consistent sign
        # transform these quats into appropriate model inputs
        # reload(g)
        reload(Rot)
        device = "cuda"
        data = g.gen_featured_data(qs_t, trainer.config['training_config'])
    else:
        data = torch.load(configs['training_config']['data_save_path'])
        data.data, data.labels = data.data[:, ...], data.labels[:, ...]
        qs_t = data.labels.cpu().numpy()
        n_ts = qs_t.shape[0]
        print("Loaded data from file")

    if not os.path.isfile(activation_path) or gen_test_data:
        network_model = trainer.model.to(device)
        network_model.eval()

        test_dataloader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=False)
        for i, (features, labels) in enumerate(test_dataloader):
            features = features.to(device)
            labels = labels.to(device)

            def get_hidden_state(module, input, output):
                global hidden_states
                hidden_states.append(output[0].detach().clone())

            network_model.rnn.register_forward_hook(get_hidden_state)
            hidden_states = []
            out = network_model(features)
            index = 49
            if i == 0:
                Y = hidden_states[0][:, index, :].cpu().numpy().T
            else:
                Y = np.concatenate((Y, hidden_states[0][:, index, :].cpu().numpy().T))
            pickle.dump(Y, open(activation_path, "wb"))
    else:
        Y = pickle.load(open(activation_path, "rb"))
        if len(Y.shape) == 3:
            Y = Y[0, ...]
        print("Loaded activations from file")
        points = Y.shape[1]

    if noise:
        qs_t[..., :1] += np.random.normal(0, 0.1, qs_t[..., :1].shape)
        qs_t[..., :] = np.sign(qs_t[..., :1]) * qs_t[..., :]  # consistent sign
        qs_t = qs_t / np.sqrt(np.sum(qs_t ** 2, axis=1, keepdims=True))  # normalize
    theta = 2 * np.arccos(qs_t[..., :1])
    u = qs_t[..., 1:]
    xs = 0.5 * u * theta

    print('\n\ntuning curves:')

    plt.figure(figsize=(15, 15 * n_size / 16))
    for i in range(n_size):
        # mean, std = [arr.cpu().detach().numpy() for arr in [fmean, fstd]]
        # mean = mean[0, 0, :, :]
        mean = Y[:, :]
        ax = plt.subplot(int(n_size / 4), 4, i + 1, projection='3d')
        n = i
        ax.scatter3D(xs[:, 0], xs[:, 1], xs[:, 2], c=mean[n, :], cmap='coolwarm', alpha=0.5, s=4)
        ax.set_xticks([]);
        ax.set_yticks([]);
        ax.set_zticks([])
    plt.tight_layout()
    plt.savefig(configs['save_path'] + "\\best_model_tuning_curves_Ground_Truth.png")
    plt.close()

    # Assuming you have already defined xs, fmean, and fstd appropriately

    # Extract mean values
    mean = Y[:, :]

    # Create a figure with subplots
    fig = go.Figure()
    for i in range(n_size):
        n = i
        fig.add_trace(go.Scatter3d(
            x=xs[:, 0],
            y=xs[:, 1],
            z=xs[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                reversescale=True,
                color=mean[n, :],
                opacity=0.5,
                colorbar=dict(
                    title='Mean Value',
                    tickvals=[mean[n, :].min(), 0, mean[n, :].max()],
                    ticktext=[f"Low: {min(mean[n, :]):.3g}", f"Medium: {0}", f"High: {max(mean[n, :]):.3g}"],
                ),
                colorbar_title='Mean Value',
                colorscale="rdbu"
            ),
            name=f'Tuning Curve {i + 1}',
            showlegend=False,
            visible=False,
        ))

    # Update layout for each subplot
    fig.update_layout(
        height=800,  # Adjust height to match your matplotlib figsize
        width=800,  # Adjust width to match your matplotlib figsize
        title='Tuning Curves',
        legend=dict(
            title='Tuning Curves',
            x=0.7,
            y=1.2
        ),
        scene=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        scene2=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        scene3=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        scene4=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        scene5=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        scene6=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        scene7=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        )
    )
    fig.data[0].visible = True
    # Create update menu buttons
    updatemenus = [
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label=f'Tuning Curve {i + 1}',
                     method='update',
                     args=[{'visible': [j == i for j in range(8)]}])
                for i in range(8)
            ],
        )
    ]

    # Update layout for each subplot
    fig.update_layout(
        updatemenus=updatemenus,
        height=800,  # Adjust height to match your matplotlib figsize
        width=800,  # Adjust width to match your matplotlib figsize
        title='Tuning Curves',
        legend=dict(
            title='Tuning Curves',
            x=0.7,
            y=1.2
        ),
        scene=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        )
    )
    pickle.dump(fig, open(configs['save_path'] + "\\best_model_tuning_curves_Ground_Truth.pkl", "wb"))
    if show:
        fig.show()


def plot_tuning_curves(trainer, configs, n_size, activation_path, gen_test_data=True,
                       get_hidden_state=get_hidden_state,
                       show=False):
    global hidden_states
    device = mgp.utils.get_device()
    n_ts = 800  # total time points

    if not os.path.exists(configs['training_config']['data_save_path']) or gen_test_data:
        # generate quaternions over time using an autoregressive process
        qs_t = np.random.normal(0, 1, (n_ts, 4))  # random points on sphere
        qs_t = qs_t / np.sqrt(np.sum(qs_t ** 2, axis=1, keepdims=True))  # normalize
        for t in range(1, n_ts):  # autoregressive process
            step = np.random.normal(0, 0.4, 3)  # displacement in tangent space
            theta = np.sqrt(np.sum(step ** 2))
            v = step / theta
            qt = np.concatenate(
                [np.ones(1) * np.cos(theta), np.sin(theta) * v])  # displacement in quaternion space
            qs_t[t, :] = Rot.q_mult_np(qt, qs_t[t - 1, :])
            # print(qs_t[t, :], np.sum(qs_t[t, :]**2), theta)
        qs_t = np.sign(qs_t[:, :1]) * qs_t  # consistent sign

        dt_dists = 4 * (1 - np.sum(qs_t[:-1, :] * qs_t[1:, :], axis=1) ** 2)
        print('consecutive displacements:', np.quantile(dt_dists, [0.1, 0.25, 0.5, 0.75, 0.9]))
        reload(g)
        reload(Rot)
        device = "cuda"
        data0 = g.gen_featured_data(qs_t, trainer.config['training_config'])
    else:
        data0 = torch.load(configs['training_config']['data_save_path'])
        qs_t = data0.labels.cpu().numpy().T
        n_ts = qs_t.shape[0]
        n_ts1 = n_ts // 2
        n_neurons = qs_t.shape[1]
        print("Loaded data from file")
    if not os.path.isfile(activation_path) or gen_test_data:
        features, labels = data0.data, data0.labels
        features = features.to(device)
        labels = labels.to(device)
        trainer.model = trainer.model.to(device)
        trainer.model.eval()

        trainer.model.rnn.register_forward_hook(get_hidden_state)
        torch.set_default_dtype(torch.float32)
        try:
            out, loss = trainer.forward(features, labels)
        except:
            out, loss = trainer.forward(torch.tensor(features, dtype=torch.float32).to(device),
                                        torch.tensor(labels, dtype=
                                        torch.float32).to(device))

        out_loss = torch.utils.data.TensorDataset(out, loss)
        torch.save(out_loss, configs['save_path'] + "\\best_model_out_loss.pth")

        torch.set_default_dtype(torch.float64)
        index = 49
        Y = hidden_states[0][:, index, :].cpu().numpy().T
        if len(Y.shape) == 2:
            Y = np.array([Y])
        Y = np.array(Y, dtype=np.double)
        pickle.dump(Y, open(activation_path, "wb"))
    else:
        Y = pickle.load(open(activation_path, "rb"))
        if len(Y.shape) == 2:
            Y = np.array([Y])
        Y = np.array(Y, dtype=np.double)
        print("Loaded activations from file")
    _, n, m = Y.shape
    print(f"Y shape: {Y.shape}")
    d = 1  # dims of latent space
    n_z = 15  # number of inducing points
    n_samples = 1  # number of samples

    if not os.path.exists(configs['save_path'] + "\\best_model_mgplvm.pkl") or gen_test_data:
        def build_model():
            # specify manifold, kernel and rdist
            manif = mgp.manifolds.So3(m, d)  # latent distribution manifold
            lat_dist = mgp.rdist.ReLie(manif, m, n_samples)  # construct ReLie distribution
            # Note: we construct the kernel and likelihood by passing the data in for initialization
            kernel = mgp.kernels.QuadExp(
                n, manif.distance
            )  # Use an exponential quadratic (RBF) kernel
            lik = mgp.likelihoods.Gaussian(n)  # Gaussian likelihood
            # lprior = mgp.lpriors.Uniform(manif)  # Prior on the manifold distribution
            lprior = mgp.lpriors.Brownian(manif, fixed_brownian_c=True, fixed_brownian_eta=False,
                                          brownian_eta=torch.ones(d) * 2 ** 2)
            z = manif.inducing_points(n, n_z)  # build inducing points
            model = mgp.models.SvgpLvm(
                n, m, n_samples, z, kernel, lik, lat_dist, lprior, whiten=False  # True
            ).to(device)
            return model

        data = torch.tensor(Y, device=device, dtype=torch.float32)
        model = build_model()

        train_params = mgp.crossval.training_params(max_steps=1500, n_mc=16, lrate=5e-2, print_every=100,
                                                    burnin=50,
                                                    analytic_kl=1)
        progress = mgp.crossval.train_model(model, data, train_params)  # train model
        pickle.dump(model, open(configs['save_path'] + "\\best_model_mgplvm.pkl", "wb"))

    else:
        model = pickle.load(open(configs['save_path'] + "\\best_model_mgplvm.pkl", "rb"))

    n_ts = 1600  # total time points

    # generate quaternions over time using an autoregressive process
    qs_t = np.random.normal(0, 1, (n_ts, 4))  # random points on sphere
    qs_t = qs_t / np.sqrt(np.sum(qs_t ** 2, axis=1, keepdims=True))  # normalize
    for t in range(1, n_ts):  # autoregressive process
        step = np.random.normal(0, 0.4, 3)  # displacement in tangent space
        theta = np.sqrt(np.sum(step ** 2))
        v = step / theta
        qt = np.concatenate([np.ones(1) * np.cos(theta), np.sin(theta) * v])  # displacement in quaternion space
        qs_t[t, :] = Rot.q_mult_np(qt, qs_t[t - 1, :])

    qs_t = np.sign(qs_t[:, :1]) * qs_t  # consistent sign

    theta = 2 * np.arccos(qs_t[..., :1])
    u = qs_t[..., 1:]

    xs = (0.5 *
          u * theta)

    query = torch.tensor(qs_t.T,
                         dtype=torch.get_default_dtype(),
                         device=device)[None, None, ...]
    data = torch.tensor(Y, dtype=torch.get_default_dtype()).to(device)
    fmean, fvar = model.obs.predict(query, full_cov=False)
    fstd = fvar.sqrt()
    print('\n\ntuning curves:')

    plt.figure(figsize=(15, 15 * n_size / 16))
    for i in range(n_size):
        mean, std = [arr.cpu().detach().numpy() for arr in [fmean, fstd]]
        mean = mean[0, 0, :, :]
        ax = plt.subplot(int(n_size / 4), 4, i + 1, projection='3d')
        n = i
        ax.scatter3D(xs[:, 0], xs[:, 1], xs[:, 2], c=mean[n, :], cmap='coolwarm', alpha=0.5, s=4)
        ax.set_xticks([]);
        ax.set_yticks([]);
        ax.set_zticks([])
    plt.tight_layout()
    plt.savefig(configs['save_path'] + "\\best_model_tuning_curves_mgplvm.png")
    plt.close()

    # Assuming you have already defined xs, fmean, and fstd appropriately

    # Extract mean values
    mean = fmean.cpu().detach().numpy()[0, 0, :, :]

    # Create a figure with subplots
    fig = go.Figure()
    for i in range(n_size):
        n = i
        fig.add_trace(go.Scatter3d(
            x=xs[:, 0],
            y=xs[:, 1],
            z=xs[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                reversescale=True,
                color=mean[n, :],
                opacity=0.8,
                colorbar=dict(
                    title='Mean Value',
                    tickvals=[mean[n, :].min(), 0, mean[n, :].max()],
                    ticktext=[f"Low: {min(mean[n, :]):.3g}", f"Medium: {0}", f"High: {max(mean[n, :]):.3g}"],
                ),
                colorbar_title='Mean Value',
                colorscale="rdbu"
            ),
            name=f'Tuning Curve {i + 1}',
            showlegend=False,
            visible=False,
        ))

    # Update layout for each subplot
    fig.update_layout(
        height=800,  # Adjust height to match your matplotlib figsize
        width=800,  # Adjust width to match your matplotlib figsize
        title='Tuning Curves',
        legend=dict(
            title='Tuning Curves',
            x=0.7,
            y=1.2
        ),
        scene=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        scene2=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        scene3=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        scene4=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        scene5=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        scene6=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        scene7=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        )
    )
    fig.data[0].visible = True
    # Create update menu buttons
    updatemenus = [
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label=f'Tuning Curve {i + 1}',
                     method='update',
                     args=[{'visible': [j == i for j in range(8)]}])
                for i in range(8)
            ],
        )
    ]

    # Update layout for each subplot
    fig.update_layout(
        updatemenus=updatemenus,
        height=800,  # Adjust height to match your matplotlib figsize
        width=800,  # Adjust width to match your matplotlib figsize
        title='Tuning Curves',
        legend=dict(
            title='Tuning Curves',
            x=0.7,
            y=1.2
        ),
        scene=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        )
    )
    pickle.dump(fig, open(configs['save_path'] + "\\best_model_tuning_curves_mgplvm.pkl", "wb"))
    if show:
        fig.show()

    n_ts = 1600

    if not os.path.isfile(configs['training_config']['data_save_path']) or gen_test_data:
        # generate quaternions over time using an autoregressive process
        qs_t = np.random.normal(0, 1, (n_ts, 4))  # random points on sphere
        qs_t = qs_t / np.sqrt(np.sum(qs_t ** 2, axis=1, keepdims=True))  # normalize
        for t in range(1, n_ts):  # autoregressive process
            step = np.random.normal(0, 0.4, 3)  # displacement in tangent space
            theta = np.sqrt(np.sum(step ** 2))
            v = step / theta
            qt = np.concatenate(
                [np.ones(1) * np.cos(theta), np.sin(theta) * v])  # displacement in quaternion space
            qs_t[t, :] = Rot.q_mult_np(qt, qs_t[t - 1, :])

        qs_t = np.sign(qs_t[:, :1]) * qs_t  # consistent sign
        # transform these quats into appropriate model inputs
        # reload(g)
        reload(Rot)
        device = "cuda"
        data = g.gen_featured_data(qs_t, trainer.config['training_config'])
    else:
        data = torch.load(configs['training_config']['data_save_path'])
        data.data, data.labels = data.data[:, ...], data.labels[:, ...]
        qs_t = data.labels.cpu().numpy()
        n_ts = qs_t.shape[0]
        print("Loaded data from file")

    if not os.path.isfile(activation_path) or gen_test_data:

        network_model = trainer.model.to(device)
        network_model.eval()

        test_dataloader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=False)
        for i, (features, labels) in enumerate(test_dataloader):
            features = features.to(device)
            labels = labels.to(device)

            def get_hidden_state(module, input, output):
                global hidden_states
                hidden_states.append(output[0].detach().clone())

            network_model.rnn.register_forward_hook(get_hidden_state)
            hidden_states = []
            out = network_model(features)
            index = 49
            if i == 0:
                Y = hidden_states[0][:, index, :].cpu().numpy().T
            else:
                Y = np.concatenate((Y, hidden_states[0][:, index, :].cpu().numpy().T))
            pickle.dump(Y, open(activation_path, "wb"))
    else:
        Y = pickle.load(open(activation_path, "rb"))
        if len(Y.shape) == 3:
            Y = Y[0, ...]
        print("Loaded activations from file")
        points = Y.shape[1]

    theta = 2 * np.arccos(qs_t[..., :1])
    u = qs_t[..., 1:]
    xs = 0.5 * u * theta

    print('\n\ntuning curves:')

    plt.figure(figsize=(15, 15 * n_size / 16))
    for i in range(n_size):
        # mean, std = [arr.cpu().detach().numpy() for arr in [fmean, fstd]]
        # mean = mean[0, 0, :, :]
        mean = Y[:, :]
        ax = plt.subplot(int(n_size / 4), 4, i + 1, projection='3d')
        n = i
        ax.scatter3D(xs[:, 0], xs[:, 1], xs[:, 2], c=mean[n, :], cmap='coolwarm', alpha=0.5, s=4)
        ax.set_xticks([]);
        ax.set_yticks([]);
        ax.set_zticks([])
    plt.tight_layout()
    plt.savefig(configs['save_path'] + "\\best_model_tuning_curves_Ground_Truth.png")
    plt.close()

    # Assuming you have already defined xs, fmean, and fstd appropriately

    # Extract mean values
    mean = Y[:, :]

    # Create a figure with subplots
    fig = go.Figure()
    for i in range(n_size):
        n = i
        fig.add_trace(go.Scatter3d(
            x=xs[:, 0],
            y=xs[:, 1],
            z=xs[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                reversescale=True,
                color=mean[n, :],
                opacity=0.5,
                colorbar=dict(
                    title='Mean Value',
                    tickvals=[mean[n, :].min(), 0, mean[n, :].max()],
                    ticktext=[f"Low: {min(mean[n, :]):.3g}", f"Medium: {0}", f"High: {max(mean[n, :]):.3g}"],
                ),
                colorbar_title='Mean Value',
                colorscale="rdbu"
            ),
            name=f'Tuning Curve {i + 1}',
            showlegend=False,
            visible=False,
        ))

    # Update layout for each subplot
    fig.update_layout(
        height=800,  # Adjust height to match your matplotlib figsize
        width=800,  # Adjust width to match your matplotlib figsize
        title='Tuning Curves',
        legend=dict(
            title='Tuning Curves',
            x=0.7,
            y=1.2
        ),
        scene=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        scene2=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        scene3=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        scene4=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        scene5=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        scene6=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        scene7=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        )
    )
    fig.data[0].visible = True
    # Create update menu buttons
    updatemenus = [
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label=f'Tuning Curve {i + 1}',
                     method='update',
                     args=[{'visible': [j == i for j in range(8)]}])
                for i in range(8)
            ],
        )
    ]

    # Update layout for each subplot
    fig.update_layout(
        updatemenus=updatemenus,
        height=800,  # Adjust height to match your matplotlib figsize
        width=800,  # Adjust width to match your matplotlib figsize
        title='Tuning Curves',
        legend=dict(
            title='Tuning Curves',
            x=0.7,
            y=1.2
        ),
        scene=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        )
    )
    pickle.dump(fig, open(configs['save_path'] + "\\best_model_tuning_curves_Ground_Truth.pkl", "wb"))
    if show:
        fig.show()


def plot_generic(xs, Y, labels, configs, show=False):
    # Y is a list of values each of whose dimension corresponding to the length of xs.
    # labels give labels to call these plots.
    print('\n\nValues:')
    if isinstance(Y, list):
        for i, y in enumerate(Y):
            if isinstance(y, torch.Tensor):
                Y[i] = y.cpu().detach().numpy()
    Y = np.array(Y)

    n_size = Y.shape[0]
    plt.figure(figsize=(15, 15 * n_size / 16))
    for i in range(n_size):
        # mean, std = [arr.cpu().detach().numpy() for arr in [fmean, fstd]]
        # mean = mean[0, 0, :, :]
        mean = Y[:, :]
        ax = plt.subplot(int(n_size / 4), 4, i + 1, projection='3d')
        n = i
        ax.scatter3D(xs[:, 0], xs[:, 1], xs[:, 2], c=mean[n, :], cmap='GnBu', alpha=0.3, s=4)

        ax.set_xticks([]);
        ax.set_yticks([]);
        ax.set_zticks([])
    plt.tight_layout()
    plt.savefig(configs['save_path'] + "\\best_model_tuning_curves_Ground_Truth.png")
    if show:
        plt.show()
    plt.close()

    # Assuming you have already defined xs, fmean, and fstd appropriately

    # Extract mean values
    mean = Y[:, :]

    # Create a figure with subplots
    fig = go.Figure()
    for i in range(n_size):
        n = i
        fig.add_trace(go.Scatter3d(
            x=xs[:, 0],
            y=xs[:, 1],
            z=xs[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                reversescale=False,
                color=mean[n, :],
                opacity=0.5,
                colorbar=dict(
                    title='Mean Value',
                    tickvals=[mean[n, :].min(), 0, mean[n, :].max()],
                    ticktext=[f"Low: {min(mean[n, :]):.3g}", f"Medium: {0}", f"High: {max(mean[n, :]):.3g}"],
                ),
                colorbar_title='Mean Value',
                colorscale="gnbu"
            ),
            name=labels[n],
            showlegend=False,
            visible=False,
        ))

    # Update layout for each subplot
    fig.update_layout(
        height=800,  # Adjust height to match your matplotlib figsize
        width=800,  # Adjust width to match your matplotlib figsize
        title='Loss values',
        legend=dict(
            title='Loss values',
            x=0.7,
            y=1.2
        ),
        scene=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        scene2=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        scene3=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        scene4=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
    )
    fig.data[0].visible = True

    # Create update menu buttons
    updatemenus = [
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label=labels[n],
                     method='update',
                     args=[{'visible': [j == n for j in range(4)]}])
                for n in range(4)
            ],
        )
    ]

    # Update layout for each subplot
    fig.update_layout(
        updatemenus=updatemenus,
        height=800,  # Adjust height to match your matplotlib figsize
        width=800,  # Adjust width to match your matplotlib figsize
        title="Loss values",
        legend=dict(
            title='Loss values',
            x=0.7,
            y=1.2
        ),
        scene=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        )
    )
    pickle.dump(fig, open(configs['save_path'] + "\\Loss_on_circle.pkl", "wb"))
    if show:
        fig.show()
