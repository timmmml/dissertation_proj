Talk about my current project
Short: 
- **Title**: _Continuous attractor reps of different manifolds in the brain_
- 1. **High-Level Motivation**: The brain use internal representations for motor control, and variables to represent live in non-Euclidean manifolds. How do neural networks learn to represent and integrate velocities on arbitrary manifolds? --> dynamical system view and CAN postulates (maintain and update)
- 2. **Background**: classical attractor networks tend to be hand-crafted based on principles such as symmetry in the weight distribution. Existing attempts to learn attractors by opmisation rely on supervision to report coordinates (inductive bias)
- 3. **Methodology**: 
    - **Architecture**: Partitioned neural network with contrastive loss
    - **Training**: Unsupervised learning with contrastive loss and temperature hyperparameter
- 4. **Results**: 
	- **Manifolds**: 1D ring, 2D torus, sphere S2S^2S2, and SO(3)
    - **Evaluation**: Dimensionality reduction, attractorness, state decodability
	- **Compare with well-known ring attractor**: symmetric weight matrix with velocity-gated assymetric connections at convergence. However, there are alternative solutions when we also constrain the network to for example represent the zero rotation by a baseline activity state. If you plug a one-layer FF controller to inject velocities to take from one activity state to another, you see controller units to tune for the current and target directions, consistent with the fly data.
- 5. **Significance**: 
	- **Sampling neural representations**: generative approach to hypothesize plausible neural representations
	- **Bridging theoretical models and empirical data**: predictions of structure and connectivity in neural circuits
- 6. **Next Directions & Potential Challenges**:
	  - **Hypothesis-Driven Modifications**: biologically plausible constraints
		  - wiring cost
		  - cell-type specificity
	  - **Self-supervised learning for state-space control** 
		  - the one-layer controller is a simplified setup to study neural signatures
		  - this framework can be extended to investigate how the brain might learn to control its internal representations within a complete motor control loop involving sensory feedback. 


PhD motivation: 
1: the field fits my interests (young interest in how animals move the way they do, gradually refined to that of figuring out how understandings of the brain may translate to engineered solutions that improves people's quality of life)
- specifically computational: i feel like this is a beautiful intersection between different neuroscience sub-fields and most directly answers my question in interest (big question, answer in small pieces)
- also readily extend-able to new methods
- super vibrant! new data, new method, best time for neuroscience

2: I chatted with friends that go on to industry and figured that the life of a PI is what i want (instead of exploiting what I have grasped in my several years of undergraduate years and become some analyst/software engineer, I want to explore more and take this chance to furnish not only my skillset but also my agency)
- Guillaume's life sounds like what I want - teaching, supervising, *deciding where the lab is going and being part of something bigger*
- Cohort of like-minded people to grow together and do hacky things together. 

3: the life of a researcher naturally fits with my strengths - I have high grits, curiosity, and preserverance, and would like to dig in every possible rabbit hole ever. 

NYU motivation
- best place to do my neuroscience (both computational and data)

- Danique Jeurissen: 
	- paper 1: reaction time on double decision task: 
		- problem: how the brain integrates multiple sources of information to make decisions
		- finding: serial instead of parallel processing in the brain
			- also distributed instead of sequential
			- seems to be some information processing bottom-neck: what may that be? attention process and winner take all? how come drift-diffution can be "here and there", then? 
	- paper 2: inactiation and restoration of function in LIP
		- ipsilateral inactivation of LIP by two methods ->  -ve effect on contralateral cells/decisions (effect stronger for timing task > RDM task)
		- restoration within sessions
		- restoration across sessions

- XJ Wang
	- expertise in computational neuroscience: interest in integrating hierarchical and distributed neural circuits with biological properties (quantified structural heterogeneity across the cortex --> macroscopic gradients --> separation of time scales)
	- "bifurcation in space" for an abrupt "critical slowing down" of neural timescale.
	- also mechanistic modeling of cognitive fucntion, such as using RNN to do cognitive tasks, rule switching &c.: very relevant for hiearhical planning!
	- also "transformation of choice outcome to action plan": train networks to do task but also recap neural dynamics in recordings to reveal recorded cell-types
	*critically, his work examines circuit mechanisms for canonical cognitive functions, and is *
	- paper 1: Liu, Wang 2024: WCST: RNN for high-order rule neurons and sensorimotor neurons: sub-cell level modulation (somatic/dendritic connections) + separation of excitatory/inhibitory neurons (experimentally based cell types + connectivity patterns): 
		- rule neurons (rule) + conjunctive neurons (combination of rules and negative feedback for rule transitions) --> WTA architecture to integrate this rule information
		- top-down rule communication: several pre-defined connection patterns
	- paper 2: Yim et al. 2019: Choice outcome --> action plan "multi-task model"(WM -> IN -> RO) to match neural types
	- paper 3: learning to learn (more on the multitasking-generalization front?)
	
- Alex Williams
	- interest in neural statistics: extraction of insights from neural data: large-scale neural recordings during modulation and learning
	- TCA as a method to deal with large-scale neural data: extend PCA to tensor data: capture variance across neurons, time, and trials (typically time and trials collapsed). now can use $\sum_{r = 1}^{R}w_{n, r}b_{t, r}a_{k, r}$ to represent $X_{n, t, k}$. Respectively these are (neuron factors, temporal factors): shared variances across trials, and (trial factors): trial-specific amplitudes of neurotemporal patterns. cortical gain control model. 
	- bayesian nonparametrics (2020 paper on PP seq which generalizes the convolutional nonnegative matrix factorization to continuous time): intuition - use K sequence motifs to decompose the data into sequential elements. use gibbs sampling to sample from the posterior distribution of sequence assignments


Field of interest
- hierarchical organisation of behavioral circuits in the brain


![[Pasted image 20250123152707.png]]

strong advisory committee: 
- good feedback on science and career development
	- meta-decisions are super important

BIg issue as a scientist: What the questions are and how to ask it?

professional dev: 
- grant writing, presentation skills
- teaching, networks (travel & retreats)
- outreach/advocacy; student council



Alex Reyes: 

barral reyes wang paper: 
- cell culture, optogenetic stimulation, measure activity propagation
- synfire mode (transient, FF, locked) vs. rate mode (transient + persistent, "more recurrent"/high density, spread sim layer)
- ![[Pasted image 20250124093600.png]]

- with simulation this is found to be dependent on number of active neurons in layers


Tudoras reyes: 
- topoligical analysis to describe *spatial patterns* in the source layer that generate suprathreshold inputs to the target layer
- simplicial complexes

	idea:
	- construct a pair of two-dimensional sheets of neuraons in feedforward configuration (source layer and target layer) (descrete units, but in this case no gap as a simplification) (let's say the target layer is organized the same as the source)
	- identify synaptic field; 
	- succefssful prop(signals) depends on the spatial rrangements of active neurons; use *simplices to describe spatial distribution of source neurons that generate firing in target neurons*
	- **capture intersections between fields in $S^{\beta}$ to capture the relationship between synaptic overlap and spatial distribution of soruce neurons and synaptic field extent**
	- check for $(n_{\theta} - 1)$ symplices in source layer to be 
	"We developed a mathematical framework for identifying spatial patterns of activity in one layer that are necessary for propagation of signals across neural networks. We constructed ˇ Cech complexes in the source layer that are constrained by the extent of synaptic fields in the target layer. By specifying the minimum number of inputs needed for a target neuron to fire, the ‘functional’ simplices and the associated source neurons that form its vertices can be determined. This proved useful for distinguishing neurons within an active population that actually contribute to signal transmission and for detecting multiple clusters that may result from complex stimuli."

Question for Alex: 
- you work a lot on the basic building blocks of the brain, relating structure to _possible function_ such as propagation of infromation through cortical layers. What would be your predictions for neuropixel datasets? How does this understanding aid in the interpretation of these data
- how about going along the cortex (so far across, as you talk about layers)
	- I am deeply intrigued by the idea of viewing the cortex as a big, folded sheet of neurons that has intra- and inter-layer structures. How do you think this can aid into capturing the canonical cortical column function? 
	- how about inter-regional connection (assumingly making these cortical functions hierachical?)
- inter-regional heterogeneity: in inputs, in cell types, etc. How do scale up your modelling framework to incorporate these? (how does the feedforward algo across layers generate the apparent feedforward, hierarchical propagation of say visual information?)


Now David Schneider: 
- Big question: how the brain learns from the past to make predictions about the future
- small angle: internal model, as done in the auditory system: predict and suppress the expected sensory input of one's on actions. 
two approaches:
- natural, ethological behavior: unrestrained rodents "social-vocal lives of mongolian gerils"
- engineered behavior (lever-press + sound association) (e.g., in the manuscript Zhou and Schneider 2024)
	- use NMF + k-means to cluster neural activity (based on participation in modes of firing) --> type 1 cells that are primarily "suppression-focused" and type 2 cells that are more "motor-expectation-focused". The latter class fire on action, and selective cells fire _more_ when the sound is expected
	- closed loop behavior + fictive pairing; silent training phase + sound training phase; two-time-point recording to see changes and modulations.  
	pre-learning: neural activity dimensions (movement and sensory) live in orthogonal spaces, and post-learnig: movement activity are aligned with expected sensory dimensions
	![[Pasted image 20250124111856.png]]

	
research projects:
- *local* circuits for internal models
- *global* networks for learning and predicting 
- *cautious walking* 
- *social-vocal decision making* (internal models about others, presumably)

*internal models used for motor control? for example if i use a tone to describe a goal state space and attach lever to a function, the internal model becomes that of the change in pitch for example; how do you think this can be reflected in the brain and achieved via some learning model? (following the paper you may suggest that somehow the auditory cortex only cares about the expected tone - but in fact the action causes an expected change in tone and how to encode that? *
- *how does this relate to the hierarchical planning  and how is this information learned (e.g., via cerebellum)*


*why audition? internal models are also widely studied in say motor control - any intersects?*



Audette and Schneider 2023: 
- prediction error paper: 
- general error response vs. specificity to errors 
- deviation from expectation in distinct dimnesions ()
- suppression to multiple dimensions simultaneously; 
- distinct population of neurons encode prediction errors; encode one/two specific violation (not generic error signal). 

Audette, Zhou, Chioma and Schneider '22
- baseline: frequency-specific supression of movement-related sounds
- unknown if the modulation is due to behaviorally-specific, timed prediction, nor whether the *expectation signals* are present in the auditory cortex. 
- L2/3, L5 --> prediction based suppression + prediction-error neurons unresponsive otherwise; recording with no sound --> movement signals in deep layers + expected signals
- populations of auditory cortical neurons to *movement, expectation, and error signals*


notes on TEM (Whittington et al. 2020): 
- generative modelling in approximate inference: 
$$
\begin{align}
q_{\phi}(\mathbf{g}_{\leq T}, \mathbf{p}_{\leq T}  \mid \mathbf{x}_{\leq T}) = \prod_{t = 1}^{T} q_{\phi}(\mathbf{g}_{t}\mid\mathbf{x}_{\leq T}, \mathbf{M}_{t - 1}, \mathbf{g}_{t - 1}, \mathbf{a}_{t})q_{\phi}(\mathbf{p}_{t} \mid \mathbf{x}_{\leq t}, \mathbf{g}_{t})
\end{align}
$$
Now $\mathbf{M}_{t} = hebbian(\mathbf{M}_{t - 1}, \mathbf{p}_{t})$
$q_{\phi}(\mathbf{g}_{t} \mid\dots) = q_{\phi}(\mathbf{g}_{t}\mid\mathbf{g}_{t - 1}, \mathbf{a}_{t})q_{\phi}(\mathbf{g}_{t}\mid x_{\leq t}, \mathbf{M}_{t - 1})$  

process is to path integrate and use sensory information
$\mathbf{M}_{t}$ as an attractor network, feed it with $\mathbf{x}_{t}$ to get the current sensorium $\mathbf{p}_{t}^{x}$ which is the memory conditioned on the sensory information only


M Long: 
- question 0: for zebra finches I think speech can be thought of as an "action", where the sequence is learned &c (which is internally a "option" under hierarchical reinforcement learning) - (keyboard type situation where I kind of see a limit cycle coming). How is this limit cycle stabilized. 
- how are these behavior altered in psychiatrical situations? network dynamics &c.?
- question direction 1: I have been thinking about how speech is planned because it is indeed quite an unusual thing: you turn thoughts, which are complex sequences of concepts, into motor sequences, in the hope of conveying this set of meanings forward. What do you think underlies the planning of speech? (decode intentions into symbols &c.) and how does this relate to the model animals you work with? 
	- I think this use of contextual information can be useful in speech prosthetics, no? 
- question direction 2: what would be your research directions in the next few years and how do you plan your research projects in the several model organisms for answering these questions? 
- question direction 3: how do you manage such a huge lab and how do hospital labs work




Hannah Payne: 
2021 science
- place cells in titmice, tuned to positions 225ms into the future, some also display head directionand speed tuning (but not all); hence display features of stuff happening in mammals
- anatomical organization: anterior-posterior axis organizes spatial information and stability, but not along the other stereotaxis axes. (how exactly is this organization? spatial location-Hc location or general tuning preference?)
- ethological demands/expereince: difference in spatial coding between ttm and songbird: how is spatial coding different? ans: sparse coding (and formation of memory - adaptive advantage) 
- SWRs as well! (both online and offline)
- (do we find grid cells?)


Determining the sites and directions of plasticity underlying changes in neural activity and behavior is critical for understanding mechanisms of learning. Identifying such plasticity from neural recording data can be challenging due to feedback pathways that impede reasoning about cause and effect. We studied interactions between feedback, neural activity, and plasticity in the context of a closed-loop motor learning task for which there is disagreement about the loci and directions of plasticity: vestibulo-ocular reflex learning. We constructed a set of circuit models that differed in the strength of their recurrent feedback, from no feedback to very strong feedback. Despite these differences, each model successfully fit a large set of neural and behavioral data. However, the patterns of plasticity predicted by the models fundamentally differed, with the direction of plasticity at a key site changing from depression to potentiation as feedback strength increased. Guided by our analysis, we suggest how such models can be experimentally disambiguated. Our results address a long-standing debate regarding cerebellum-dependent motor learning, suggesting a reconciliation in which learning-related changes in the strength of synaptic inputs to Purkinje cells are compatible with seemingly oppositely directed changes in Purkinje cell spiking activity. More broadly, these results demonstrate how changes in neural activity over learning can appear to contradict the sign of the underlying plasticity when either internal feedback or feedback through the environment is present.

also how does the place map form? 


gaze tuning: 
- visual search and food task: 
	- 2 gaze strategies: lateral gaze (align one pupil with the target) and frontal gaze (when the bird dashes)
- closed loop version: light cue in response to the bird gazing at the correct set; precise control of timing and ensures that the cue is not going to be explained by peripheral vision. 

- find gaze cells, tuned to target with the contralateral eye
- strongly correlated spatial cells and gaze cells (nonrandom mix) - same neurons, overlap in space; 
- gaze response encodes internal prediction (early response during saccade andlate response after saccade, respectively doing prediction and sensory feed-in)
- **Remote Place Coding by Gaze:**
    - Place cells fired not only when the bird occupied their preferred location but also when it gazed at that location from a distance.
- **Prediction and Reaction States:**
    - Early responses encoded internal predictions about the next gaze target, while late responses encoded visual input from the current target.
- **Interneuron Dynamics:**
    - Different classes of inhibitory neurons (classified as **Peak** and **Trough**) fired at distinct phases of the saccadic cycle, suggesting a role in coordinating excitatory activity during visual search.
	(the saccade being the time-keeper)
### **Future Directions**

- **Neurophysiological Studies:**
    - Investigate how hippocampal circuits in other species (e.g., rodents) represent remote gaze targets.
    - Explore whether hippocampal interneurons are homologous across birds and mammals.
- **Behavioral Generalization:**
    - Examine gaze-driven hippocampal activity during more complex tasks (e.g., foraging or predator avoidance).
- **Neural Mechanisms:**
    - Investigate how visual and motor pathways contribute to gaze-driven hippocampal responses.

### **Broader Implications**

- This work connects hippocampal place coding to visual attention, suggesting that place cells generalize across **local** and **remote spatial functions**.
- The findings support theories of the hippocampus as a **spatial memory hub**, unifying:
    - Local navigation (storing physical location).
    - Remote planning (recalling distant targets via gaze).