Below is a structured set of talking points that you might incorporate into your verbal presentation and use as a basis for further brainstorming. This is aimed to help you:

- Summarize what you have done so far in your dissertation project.
- Demonstrate depth of thinking with respect to your project’s significance.
- Outline next steps and potential impact in neuroscience and related fields.

---

## 1. Project Overview

**Title**: _Continuous attractor reps of different manifolds in the brain_

**High-Level Motivation**

- In the brain, behavioural variables can naturally lie on diverse manifolds (1D ring, 2D torus, sphere, SO3, etc.). *and we want internal reps of them*
- Continuous attractor networks (CANs) are thought to be mechanistic circuit models for how neural populations maintain and update internal representations of external variables.
- Most existing methods impose strong supervised constraints (e.g., explicit path-integration tasks). Your project relaxes these assumptions, investigating how neural circuits can self-organize representations of these manifolds in an unsupervised, contrastive learning setting.

**Key Question**

- _How do neural networks, with minimal external supervision, learn to represent and integrate velocities on arbitrary manifolds?_

---

## 2. Methodological Details

### 2.1 Architecture Choice & Rationale

(RNN doesn't just "work")
- **Partitioned Neural Network**: Inspired by the known circuit motifs in the Drosophila ellipsoid body (the “shift-and-offset” mechanism).
    
    - A subnetwork processes velocity inputs (tangent vector on the manifold) and modulates the representation subnetwork.
    - This setup handles the nontrivial dependence of velocity signals on current manifold position (e.g., local tangent space changes as you move around a sphere vs. a ring).
- **Contrastive Loss with Tunable Temperature**:
    
    - You impose that similar “positions” on the manifold should have similar neural representations, while dissimilar ones should be farther apart.
    - The temperature hyperparameter can adjust how “sharp” the similarity constraints are.

### 2.2 Training and Evaluations

1. **Manifolds Studied**
    
    - 1D ring, 2D torus, sphere S2S^2S2, and SO(3).
    - The architecture is flexible enough to handle different manifold topologies.
2. **Validation Approaches**
    
    - **Dimensionality Reduction**: Visualize the learned neural activity and check if it embeds the target manifold structure (e.g., PCA, t-SNE, UMAP).
    - **Attractorness**: Quantify how stable the representation is by measuring gradient norms vs. noise-induced perturbations. A “true” continuous attractor should resist small perturbations yet remain continuous over the manifold.
    - **State Decodability**: Train decoders to read out the manifold coordinates from the neural population, confirming that the representation faithfully encodes position on the manifold.
3. **Activation Function Exploration**
    
    - **tanh⁡\tanhtanh**: Encourages smoother, continuous tuning functions but can be less biologically realistic (negative values).
    - **Bounded ReLU**: More biologically plausible. Tends to form multiple stable attractors (sometimes collapses to discrete states) if not carefully tuned. This offers insight into how non-linearity affects stability.
(discrete solution problem is a bit more prevalent than I thought...)

---

## 3. Control Problem Extension

- **Goal**: Use a similar circuit-level approach to move the internal representation from the current state to a target (reference) state by injecting velocity inputs.
- **Significance**:
    - Extends the idea beyond static representation to _dynamical control_ over representations.
    - Preliminary findings indicate that a simple one-layer feedforward controller can exhibit realistic joint tuning to current and target directions (paralleling neural data from flies).
    - This framework can be scaled up to more complex manifolds and more realistic controllers (e.g., multi-layer networks, neuromorphic hardware).

---

## 4. Significance and Broader Implications

### 4.1 Justification of Significance

1. **Sampling Neural Representations**
    
    - _Stochastic optimization + minimal constraints_ →\rightarrow→ multiple possible solutions.
    - Yet each solution still satisfies the same functional objectives (self-consistency, contrastive similarity).
    - This methodology thus provides a _generative approach_ to hypothesize plausible neural representations.
    - Researchers can incorporate domain-inspired constraints (e.g., wiring cost, cell-type specificity) to see how the resulting solutions differ.
2. **Bridging Theoretical Models and Empirical Data**
    
    - Provides predictions of structure and connectivity in neural circuits: e.g., predicted weight distributions, tuning functions, or correlations in neural population activity.
    - Empirical labs can look for or rule out such patterns in recorded neural data (e.g., connectomic data or large-scale recordings).
3. **Extensibility to Chained Controllers and Motor Circuits**
    
    - The same viewpoint—neural circuits representing states on manifolds, updated by velocity inputs—can in principle be stacked or chained to model hierarchical control (e.g., basal ganglia to motor cortex loops, cerebellum corrections).
    - Could highlight how multi-layer or multi-region networks coordinate to move across “manifold states.”
    - Potentially unify local attractor population models (e.g., for heading direction) with more global, goal-directed circuits.
4. **Brain-Machine Interface (BMI) Potential**
    
    - BMIs aim to decode continuous variables from neural activity (e.g., position of a limb in space).
    - Understanding how neural systems _naturally_ encode continuous spaces could inform better BMI decoders, or even improved training paradigms that align with the brain’s own manifold-based representations.

---

## 5. Next Directions & Potential Challenges

1. **Hypothesis-Driven Modifications**
    
    - Add biologically plausible constraints (e.g., sparse connectivity, Dale’s principle of excitatory/inhibitory neurons) and see how solutions change.
    - Explore multi-scale attractor networks: from local circuit modules to distributed cortical networks.
2. **Exploring Larger or More Complex Manifolds**
    
    - Go beyond SO(3) to high-dimensional configuration spaces (e.g., joint angles in limbs).
    - Investigate how the network might partition or factorize these spaces (e.g., product manifolds, Riemannian submanifolds).
3. **Hierarchical Control Architectures**
    
    - Incorporate multiple attractor networks in a layered structure, each responsible for sub-tasks or sub-manifolds.
    - Investigate how these layers coordinate to achieve a global, goal-directed behavior (chained controllers).
4. **Experimental Matching**
    
    - Propose specific neural signatures that would confirm or falsify your model in real neural data.
    - For instance, in the fly ring attractor, test predictions about connectivity motifs (shift-and-offset) or neural population patterns during velocity inputs in broader or multi-lobed structures (e.g., the mushroom body).
5. **Robustness and Stability**
    
    - Investigate the stability of solutions when facing real-world-like noise, changing inputs, or parameter drift.
    - This might highlight regime shifts between continuous vs. discrete attractors.
6. **Potential Pitfalls**
    
    - The manifold representation could partially collapse if the hyperparameters (e.g., temperature, learning rates) are not well-chosen.
    - Real neuronal circuits might have additional constraints (e.g., metabolic cost, precise spike timing) that are not well-modeled in your approach.
    - Translating from continuous attractor states to real physical movements (in robotics or BMIs) can bring about additional complexities—like feedback delays and actuator constraints.

---

## 6. Conclusion

You are justified in stating that your approach:

1. _Generates candidate neural representations_ of various manifolds without heavily engineered constraints.
2. _Proposes a framework_ for experimental design: neuroscientists can incorporate additional constraints, discover new representations, and check them against empirical recordings.
3. _Extends gracefully_ into control scenarios by injecting velocity inputs—an idea that could unify multiple aspects of neural coding and neural motor control.

To further strengthen your significance claim, you could:

- Emphasize how easily new constraints (e.g., connectivity motifs seen in biology) can be plugged into your approach to see how the emergent representations adapt.
- Highlight the potential for your framework to serve as a “model-based generator of hypotheses” for how neural circuits might encode and transition among states, all the way from ring attractors to high-dimensional motor commands.

---
Interest
- aside from being the best program in the world, Gatsby attracts me in several practical aspects: 
	- dedication to invest in the students - one year of advanced coursework to allow a common language and a shared understanding of the field among the cohort of students (which I think is very important!)
	- tight-knitted community between ML and neuroscience just like CBL but more mutual understanding from point 1. 
	- Specifically for committing to SWC-GCNU joiat program, I chose this because I want to be involved in both empirical and theoretical sides of neuroscience research and become a PI in the future. This program will give me a solid foundation in both fields, by knowing the model-system, have a mental model of how to query them, what data are available at what costs (also more direct exposure to discoveries and the physical brain). such that I can come to my laptop with the right problems to solve in mind.


Questions for Maneesh:
1. 
- what would you say be the biggest challenge in extending say GP methods to multi-regional neural data? 
 - you mentioned in one of your talks (Kavli institute 2023) that to understand, say in the visual hierarchy, how information is factorized, one way is to investigate the series of symmetry-breaking that lead to a particular hierarchical structure. What would you say determines the symmetry-breaking in the brain? Has your lab been working on this?
 - Do you think this characterization can be extended into the motor hierarchy? I can imagine a workflow of 1. determining how the brain learns to factorize actions into hierarchical motor programs in skill learning, crystalize principles of this factorization and 2. apply these principles to learn state space models for neural decoding. (constrain the latent dynamical system to be hierarchical for decoding multi-level neural recordings)

2. supervising style
- what does being supervised by you look like in action? 
- what would the project-generation process look like? 
- I am a quite autonomous student and have worked on one-person projects for most of my undergrads. Though, I would be thrilled to join in collaborative projects that are more multi-faceted.  