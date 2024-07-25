steps foward (ranked from near- to far-term): 

Plan: 
 - tackle Model 1 first; the performance seems to be limited by the encoder's ability to infer a rotation from image. This will be possible once the model is overtrained on the object. The first arrow of *image -> inferred rotation representation -> motor plan* looks very solvable if an unlimited amount of rotated cows are provided. 
 - though - this will not be robust
	 - zero-shot rotation is impossible; (in humans, it probably goes like 3D-shape inference, then rotation inference, then motor plan. *An actionable 3D-shape representation involves the ability to infer consequence of any rotation*)
	 - errors along the motion trajectory will not be monitored (no space for feedback)
	 - this strategy probably won't solve the pair-image task (rotate configuration A to match that of B)
- Models along the lines of 2-5 will involve this rotation-consequence inference, and can be more robust/plausible. They extend naturally to pair-images.

If Model 1 turns out unsuccessful, that's great - it means that you would need this forward model to solve even task0. 

If Model 1 turns out successful (more likely than not given infinite training material), focus on dissecting its differences in representations from the forward model. 

## Model 1: basic rotation model from image: 

Let $I^{*}$ be  the true rotated image; $\{ \boldsymbol{\omega}_t \}_{t \in [T_a, T_e]}$ be the series of output angular displacement in the action phase. $I^{*} =\mathrm{rotate}(q^{*}, I_0)$.
$$
\begin{align}
\boldsymbol{\omega}_{T_a:T_e}  & = f_\theta(I^{*})
\end{align}
$$
 - Optimise on geodesics & activity 

Train $f$ directly (merging pretrained pieces, perhaps a CNN-RNN structure (in lieu of recent works on MLP for vision modelling)) - try to exhaust the inference capability on this task. 

*Note: use an infinite stream of data to train here - also retrain the pretrain task with infinite data*


## Model 2: add forward component to the above model: 

$$
\begin{align}
\boldsymbol{\omega}_{T_a, T_e}, \hat{I}  & = f_{\theta}(I^{*}) \\
I' & = \mathrm{rotate}(\boldsymbol{\omega}_{T_a:T_e}, I_0) \\
C(\hat{I}, I^{*})  & = D(\hat{I}, I')
\end{align}
$$

- Optimise on geodesics & activity, but also reconstruction *of the rotation made by the network*

Caveat: $I'$ comes from a rendering process that also depends on $\theta$
- Pytorch3D rendering can be made differentiable.
- Or we can ignore gradients from the true image?

### If successful, can analyse representations

## Model 3: expand model 2 to two noninteracting neworks

Let $G$ be the motor generator; $M$ be the predictive process
Let $\mathbf{u}$ be an input stream of motor babbling
$$
\begin{align}
\boldsymbol{\omega}_{T_a, T_e}  & = G_{\theta_{G}}(\mathbf{u}) \\
I'  & = \mathrm{rotate}(\boldsymbol{\omega}_{T_a:T_e}, I_0) \\
\hat{I}  & = M_{\theta_{M}}(\mathbf{u}) \\
\end{align}
$$

Can as a next step treat this as a feedback controller to set inputs to the network. 
$$
\begin{align}
\mathbf{u}  & = K_{\theta_{K}}(I^{*}, \hat{I})
\end{align}
$$
### May serve as a pre-training step to model 2

## Model 4: share $G$ and $M$'s states;

- Motor generator decodes evolution of the RNN representation.
- Predictive model predicts the result of this generation process by initial state.
$$
\begin{align}
\mathbf{z}_{t+1}  & = f_{\theta_f}(\mathbf{z}_{t}, \mathbf{u}_{t}) \\
\text{saturate at  } & T_{a} \\
\boldsymbol{\omega}_{t}  & = G_{\theta_{G}}(\mathbf{z}_{t}), \mathbf{u}_{t} = 0 & t \in [T_a, T_e] \\
\hat{I}  & = M_{\theta_{M}}(\mathbf{z}_{T_a})
\end{align}
$$

### May serve as a pre-training step to model 2

## Model 5: Forward model, joint motor-perception hierarchy (may be a different project altogether)

Let $\mathbf{M} = \{ M_i \}_{i \in \{ 1, \dots, n \}}$; in this case we have a hierarchical model to decode motor consequence imagination.

Let $\mathbf{P} = \{ P_{i} \}_{i\in \{ 1, \dots , n \}}$ be a set of n perceptive layers

**MODIFY WHEN TIME COMES NEAR**

$$
\begin{align}
\boldsymbol{\omega}_t & = G(\mathbf{z}_t, \mathbf{f}_{1, t}) \\
\mathbf{f}_{1, t} & = M_{1}(\mathbf{f}_{1, t - 1}, \mathbf{z}_{t}, \mathbf{f}_{2, t - 1}) & \mathbf{f}_{1, t} \sim s_{1} =S_{1}(I', s_{2}) \\
\mathbf{f}_{2, t} & = M_{2}(\mathbf{f}_{2, t - 1}, \mathbf{f}_{1, t}, \mathbf{f}_{3, t - 1}) & \mathbf{f}_{2, t} \sim s_{2} = S_{2}(I', s_{1}, s_{3}) \\
\vdots\\
\dots  &  & \mathbf{f}_{n, t} \sim I'
\end{align}
$$


The above sequence of $M_{i}$ marks a hierarchy from abstract to coarse (think filter banks)

