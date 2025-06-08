### Abstract

A cognitive map has long been the dominant metaphor for hippocampal function, embracing the idea that place cells encode a geometric representation of space. However, evidence for predictive coding, reward sensitivity and policy dependence in place cells suggests that the representation is not purely spatial. We approach this puzzle from a reinforcement learning perspective: what kind of spatial representation is most useful for maximizing future reward? 

**We show that the answer takes the form of a predictive representation. This representation captures many aspects of place cell responses that fall outside the traditional view of a cognitive map. Furthermore, we argue that entorhinal grid cells encode a low-dimensionality basis set for the predictive representation, useful for suppressing noise in predictions and extracting multiscale structure for hierarchical planning.**

#### Core Computational Framework:

1. **Successor Representation (SR):**
    
    - The SR represents the expected discounted future occupancy of states starting from a given state under a particular policy.
    - Mathematically, for a state $s$, the SR matrix $M$ encodes $M(s,s′)$: the discounted expected number of times the agent visits state $s′$ starting from $s$.
    - It uses a temporal difference (TD) learning update rule: $M(s,s′)←M(s,s′)+η[I(s=s′)+γM(s′,:)−M(s,:)]$
        - $η$: Learning rate. 
        - $γ$: Discount factor controlling the predictive horizon.

2. **Analytical Solution of SR:**
    
    - When the transition matrix $T$ is known, the SR can be computed as: 
    - $M=(I−γT)−1$
    - This approach reduces computational cost by avoiding iterative updates.
	
3. **Eigenvector-Based Decomposition:**
    
    - The eigendecomposition of the SR matrix provides low-dimensional representations.
    - Grid-like representations emerge from these eigenvectors, with higher eigenvalues encoding larger spatial scales.

---

#### Simulations:

1. **Task Environments:**
    
    - **Spatial Domains:** Simulated as 2D grids, where transitions between states were defined by adjacency in the lattice.
    - **Random Walk Policy:** Used to compute unbiased SRs, assuming equal probability for all neighboring transitions.
    - **Policy Modification:** Rewards and penalties were introduced to bias transitions towards goal states or away from obstacles.
2. **Model Comparisons:**
    
    - SR place fields were compared with:
        - Gaussian place fields (Euclidean-based).
        - Geodesic place fields (shortest-path based).
    - SR fields were shown to respect topology, such as barriers and paths.
3. **Behavioral Relevance:**
    
    - Changes in SR place fields were validated against hippocampal data (e.g., Tolman detour maze experiments).
    - Backward-skewed SR place fields explained predictive coding of future states observed in hippocampal neurons during directional tasks.

---

#### Predictions and Neural Correlates:

1. **Place Cells and SR:**
    
    - Place cells encode rows of the SR matrix, predicting future occupancy given the current state.
    - SR receptive fields adapt dynamically to obstacles, goals, and changes in policy.
2. **Grid Cells as Basis Functions:**
    
    - Entorhinal grid cells were hypothesized to encode a low-dimensional eigendecomposition of the SR.
    - This decomposition regularizes noise and enables hierarchical planning by representing larger spatial scales.
3. **Subgoal Discovery:**
    
    - Bottleneck states (e.g., doorways) were identified using eigenvector-based partitions of the state space.
    - These subgoals aligned with human navigation strategies.

---

#### Computational Techniques:

1. **SR Approximation via Temporal Difference Learning:**
    
    - Incrementally updates SR using observed transitions.
2. **Eigenvector Thresholding for Grid Fields:**
    
    - Eigenvectors of the graph Laplacian were thresholded to simulate grid fields, ensuring non-negative firing rates.
3. **Simulations and Code:**
    
    - Simulations performed using MATLAB with custom algorithms.
    - Code for SR computation, eigenvector decomposition, and plotting was made available.