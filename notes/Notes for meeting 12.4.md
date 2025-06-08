Goal: figure out what's next for the project!

(first ask Guillaume how he though my presentation was)

Direction 1: formalizing the training framework; parameter tuning and experiments

- motivating question: Ila FIete's paper has used a very similar method to train grid cells, how come I don't have grid cells? 
- well, may be it's that we have over-constrained our set of nets; in their case $W(\mathbf{v}_{t}) = MLP(\mathbf{v}_{t}), \mathbf{g}_{t} = \sigma(W(\mathbf{v}_{t})\mathbf{g}_{t - 1})$; $\sigma (\cdot)=\frac{\mathrm{\mathrm{Re}LU}(\cdot) } {\mathrm{\lVert \mathrm{\mathrm{Re}LU}(\cdot) \rVert}}$

I think we don't necessarily need to venture reproducing everything they did, but it would be nice for the completeness of my thesis to at least quantify the effects of different parameters and evaluate some biological constraints. 

Defs: 
$$
\begin{align}
\{ \mathbf{v}_{t} \}_{1}^{T}, \mathbf{v}_{t} \sim_{i.i.d.} p(\mathbf{v}) \\
\{ \pi_{b} \}_{b = 1}^{B}, \pi_{b}:[T] \to [T]
\end{align}
$$ $\pi_{b}$ is a set of $B$ random permutations to apply to the sequence of velocities, then call them batches. 
$\mathbf{g}_{\pi_{b}(t)}$ is the set of neural representations; in batch $b$ and time $t$
- this is re-done for each gradient step.

Their loss function: 
- Separation (introduce $\sigma_{x}, \sigma_{g}$)
$$
\mathcal{L}_{sep} = \sum_{\lVert \mathbf{x}_{\pi_{b'}(t)} -\mathbf{x}_{\pi_{b}(t')}\rVert>\sigma_{x} }\exp\left( -\frac{\lVert \mathbf{g}_{\pi_{b'}(t)} - \mathbf{g}_{\pi_{b}(t')} \rVert^{2}}{2\sigma^{2}_{g}}  \right)
$$
- path invariance loss (uses $\sigma_{x}$ too) (probably a typo here)
$$
\mathcal{L}_{inv} = \sum_{\lVert \mathbf{x}_{\pi_{b'}(t)} - \mathbf{x}_{\pi_{b}(t')} \rVert < \sigma_{x}}\lVert \mathbf{g}_{\pi_{b}(t) } - \mathbf{g}_{\pi_{b'}(t')} \rVert ^{2}
$$

Note: the above is almost *the same* as what we did (up to a choice of similarity/distance kernel)

- *capacity loss*
$$
\mathcal{L}_{cap} = - \left\lVert   \frac{1}{BT}\sum_{\pi_{b, t}}\mathbf{g}_{\pi_{b}(t)}  \right\rVert ^{2}
$$
- *conformal isometry loss* (they might not have used it?)
$$
\mathcal{L}_{conIso} = \mathrm{Var}\left[ \left\{  \frac{\lVert \mathbf{g}_{t} - \mathbf{g}_{t - 1} \rVert }{\lVert \mathbf{v}_{t} \rVert }  \right\}_{t: 0 < \lVert \mathbf{v}_{t} \rVert < \sigma_x} \right]
$$

Commentary on the latter two: 
1. the capacity loss *encourages* large summed norms. As their stuff is positive and on the hypersphere (normalizing nonlinearity), this thing basically encourages the opposite of sparsity, that *many* neurons take part for coding *many* locations
2. the conformal isometry loss makes the magnitude of the shift in neuron space a function of the magnitude of the shift in the velocity space

Corresponding to the four losses, define $\lambda_{sep}, \lambda_{inv}, \lambda_{cap}, \lambda_{conIso}$
- note that two knobs are involved for $\mathcal{L}_{sep}$ and $\mathcal{L}_{inv}$

It seems to me that the capacity loss is the main thing
- indeed, if you remove capacity and reduce $\sigma_{g}$ (the neural length scale) then you get place cells. 
- we got all place cells

(but for a large sigma g they found grids)

more experiments [[experiments with widths]]

Anyways, I will think about this in my own time 

Direction 2 (conderably more important): 

How should we do study control among these manifold representations? 

- goal of the controller: 
$$
\mathbf{g}_{t}, \mathbf{g}_{goal} \to \mathbf{v}_{t + 1}
$$

case study in the drosophila: 
Mussells Pires et al. 2024
FC2 cells (fan shaped body columnar neurons): activity correlated to goal angle
- focal optoenetic activation --> orient along any direction as they walk forward (menotaxis)
- setup: vertical blue bar on a paranormic LED display and walk on a ball. 
	- the bar is basically like the "sun" to anchor the heading rep. 

Connections:
EPG and FC2 --> PFL3; PFL3 show spike-rate tuning to heading angle + goal angle 

Experiment 1: manipulate goal angle by jumping the bar
FC2 activity: "calcium bump that shifts across the left/right axis of the fan-shaped body""
- correlated with the bar position. 
- manipulation: open loop jump of the bar --> return to closed-loop control after 2 seconds. 
- EPG phase is locked with the manipulation; FC2 is not (the phase is irrespective of the manipulation) 

Experiment 2: manipulate FC2 optogenetically
![[Pasted image 20241204120634.png]]

Experiment 3:  virtual rotations vs. PFL3
PFL3 cells are suggested to do this heading vs. goal comparison (inputs from PB and FB, outputs to the lateral accessory lobes)
- tuning curves (both Vm and spike rates) are tuned toward heading; Vm tuning currves are even sinusoidal
also goal directed tuning: 
- for PFL3-L neurons (those that project to lLAL): reduce amplitudes when the fly's goal is to the right of the cell's preferred direction

PFL3-L:
![[Pasted image 20241204122612.png]]
(we can see this is a place cell on the 2D torus)
- the data is population averaged, across all flies and all cells. (not many goal angles are out there for each fly!)

Their model: 
single cell:
$$
g = f(\cos(H - H_{pref}) + d\cos(G - G_{pref}))
$$
(in their model fitting they assumed the same difference between the two preferences for all cells.)

circuit: 
![[Pasted image 20241204124231.png]]


Westeinde et al. 2024
![[Pasted image 20241203233333.png]]

let's say goal input and current HD are both held in "ring like" structures, and we know the alignment (anchoring by some env. stim)
- can calculate simliarity with either "shifted copy"; to each direction in the tangent space
- more similar --> L/R movement
- break symmetry by encoding gain




Direction 3: what should we do for SC? 

*an alternative hypothesis*
for the limited range of motion in SC, perhaps suffice to be a direction + local, bounded 3D space? 
- instead of the space of 3D rotations, maybe the SC codes for something else
- (similar to the eye situation: instead of containing an entire S^2  probably suffice to do a 2D space. )