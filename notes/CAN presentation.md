30 minute long presentation on the theory of the CAN 

General outline: 
- a pedagological talk on the topic of CANs
- a short presentation on my own work 

## 1. Introduction and Motivation (2–3 minutes)

- **Big-Picture Question**
    - What are continuous attractor networks (CANs)?
    - Why are they important in theoretical neuroscience?
- **Relevance to Biological Systems**
    - Stable representation of continuous variables (e.g., orientation, head direction, spatial position).
    - Linking CANs to cognitive functions: navigation, working memory, attention, etc.

**Key points to highlight**

- The concept that neural circuits can maintain persistent activity patterns corresponding to a continuous manifold of states (e.g., angle on a ring, position in 2D space).
- Why we might think of networks as supporting such a continuum of attractors (theoretical underpinnings and experimental hints).

---

## 2. Biological/Experimental Context (2–3 minutes)

- **Examples**
    - **Head-Direction Cells**: stable firing correlates with an animal’s heading.
    - **Place Cells/Grid Cells**: stable spatial tuning in the hippocampus/entorhinal cortex.
    - **Orientation Columns** in the visual cortex (simple model of orientation tuning in primary visual cortex).
- **Key Experimental Observations**
    - The “bump” of neural activity that moves continuously with changes in heading/orientation/position.
    - The robust stability of these activity profiles despite small perturbations.

**Why these contexts matter**

- They provide real biological examples where CAN models are used to interpret neural coding.
- They give insights into the functional role: e.g., path integration in navigation, maintaining a stable reference frame for orientation.

---

## 3. Classical Continuous Attractor Models (5–6 minutes)

Here, dive into the **mathematical and dynamical underpinnings**.

### 3.1 Amari’s Formulation of “Bump Attractors”

- **Foundational Works**
    - Amari (1977) introduced the idea of bump attractors with continuous neural fields.
- **Key Mathematical Idea**
    - A continuous set of neurons (or neural field) with recurrent connectivity often described by an integral/differential equation.
    - Local excitatory coupling and long-range inhibitory coupling shapes stable localized activity “bumps.”

### 3.2 Zhang (1996) Ring Attractor Model

- **The Ring Model**
    - Representing angles in a head-direction system by a one-dimensional ring of neurons.
    - Lateral connectivity ensures a stable “bump” that can translate around the ring without losing shape.
- **Mathematical Overview**
    - Typical equations: 
    $$\dot{u}(x,t)=−u(x,t)+∫W(x−y) f[u(y,t)] dy+I(x,t).$$
	
    - Explanation of how “shift invariance” leads to a continuum (rather than discrete) family of stable states.
    - Discuss the stable solution as a localized bump centered at some angle $\theta$.
    - Show how small perturbations do not break the shape but only shift its center, thus realizing a continuous manifold of attractors.

### 3.3 Stability Analysis and Key Properties

- **Attractor Manifold**
    - The ring (or line, or plane) as a continuous manifold of stable solutions.
- **Bump Width, Peak Firing Rate**
    - Typically found by analyzing self-consistency conditions (e.g., solving for stationary bump solutions).
- **Perturbation Dynamics**
    - Fluctuations lead to shift or drift on the ring rather than global destabilization.
    - Illustrates the robust nature of the attractor.

---

## 4. Extensions and Variations (3–4 minutes)

- **Ben-Yishai et al. (1995) Model** on orientation tuning in V1.
    - Local excitatory and global inhibitory network explaining orientation selectivity as a form of bump attractor.
- **Seung (1996)** on ocular fixation using continuous attractors.
    - How the brain might maintain a stable eye position despite perturbations.
- **Grid Cell Models** (Burak and Fiete, 2009)
    - Two-dimensional attractor networks that can explain hexagonal grid firing patterns.

**Key Theoretical Insights**

- Generalization from 1D to higher dimensions (rings → planes).
- The essential requirement: a shift-invariant connectivity kernel to maintain translation invariance in activity space.

---

## 5. Empirical/Experimental Validation (2–3 minutes)

- **What does it mean for biology?**
    - Single-unit electrophysiology or calcium imaging showing stable bumps in head-direction systems, orientation maps, or entorhinal grid patterns.
- **Testing Predictions**
    - CAN theories predict relationships between drift rates, global inhibition levels, and external cue alignment.
    - Neurophysiological studies that test for uniform shifts, stable tuning, and predicted attractor “landscapes.”

---

## 6. Key Open Questions and Broader Implications (2 minutes)

- **Limitations**
    - Real neural circuits are noisy, heterogeneous, and plastic.
    - Continuous attractors exist in simplified theoretical forms—how robust are they in the real brain?
- **Ongoing Directions**
    - Incorporating plasticity (Hebbian updates, homeostatic constraints).
    - Role of CANs in higher cognitive functions (working memory, decision-making).
    - The interplay of feedforward vs. recurrent inputs in shaping attractor dynamics.

---

## 7. Transition to Your Research (~30 seconds)

Use this as a **bridge** to your own work:

- **Motivate your question** based on the open issues/limitations listed above.
- **Position** your approach in the context of these classical models.

---

# References to Chase

Below are key references (both classical and more recent) that will help flesh out the mathematical and conceptual details:

1. **Zhang, K. (1996)**. _Representation of spatial orientation by the intrinsic dynamics of the head-direction cell ensemble: A theory._ **The Journal of Neuroscience**, 16(6), 2112–2126.
    
    - **Why read**: Canonical ring attractor model for head-direction cells; widely cited for fundamental CAN principles.
2. **Amari, S. (1977)**. _Dynamics of pattern formation in lateral-inhibition type neural fields._ **Biological Cybernetics**, 27, 77–87.
    
    - **Why read**: Seminal work on continuous neural field models and “bump” attractors.
3. **Ben-Yishai, R., Bar-Or, R. L., & Sompolinsky, H. (1995)**. _Theory of orientation tuning in visual cortex._ **Proceedings of the National Academy of Sciences**, 92(9), 3844–3848.
    
    - **Why read**: Shows how stable orientation tuning emerges from lateral interactions (ring attractor concept in visual cortex).
4. **Seung, H. S. (1996)**. _How the brain keeps the eyes still._ **Proceedings of the National Academy of Sciences**, 93(23), 13339–13344.
    
    - **Why read**: Another canonical model explaining maintenance of stable motor positions (continuous attractor in oculomotor system).
5. **Burak, Y., & Fiete, I. R. (2009)**. _Accurate path integration in continuous attractor network models of grid cells._ **PLoS Computational Biology**, 5(2), e1000291.
    
    - **Why read**: Extends CAN concepts to 2D for grid cells; includes noise, drift, and path-integration aspects.
6. **Tsodyks, M., & Sejnowski, T. (1995)**. _Rapid state switching in balanced cortical network models._ **Network: Computation in Neural Systems**, 6(2), 111–124.
    
    - **Why read**: Explores balanced recurrent networks that can exhibit attractor-like dynamics.
7. **Taube, J. S., Muller, R. U., & Ranck Jr., J. B. (1990)**. _Head-direction cells recorded from the postsubiculum in freely moving rats. I. Description and quantitative analysis._ **The Journal of Neuroscience**, 10(2), 420–435.
    
    - **Why read**: Pioneering experimental work identifying head-direction cells, often cited in theoretical CAN contexts.

---