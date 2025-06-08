GOALS: 

1. precisely define the problem: 
	We have now a mapping between a target rotation and some initial preparation states, which upon unrolling through a fixed LDS, produces the desired rotation. 
	
	We want to dissect this mapping to answer the question: how do the hidden states encode the rotation?. Beyond this, we aim to generalise to all optimised representations of SO3 by this class (defined by (linear) decodability into a series of actions). 

	Group representation theory tells us that we can represent the group of 3D rotations by a set of matrices, in which the group operation can be made that of matrix multiplication. Any such group representation can be decomposed into a direct sum of irreducible representations. 

- Hence, we want to ask the following questions: 
	- what's the relationship between the prep-state codes? 
		- composition of rotations (what does a group operation map to in this case)?
		explore the downstream task: make the rotation from state A to state B
		inputs: 2 images; current state, target state
		stripped down problem: operation based on a shifted ground-zero, so to perform $R_{2}R_{1}^{-1}$. How would this be computed? 
			- if the code is linear actionable, we expect the resultant acivities  to be 
		- representation of the null rotation (identity)
		- representation of inverses
		
	- what underlies the principles of the population codes? 
		- neural tuning to rotations (individual vs. population)

# Gonzalez-Rueda, Jensen et al. 2024

Idea: to assess the *functinoal alignment* between sensory and motor maps - *sensory tuning of well-defined motor units*. 

established view: sensorimtor integration in the SC results from systematic mapping of stati senosry features (spatial visualRFs/movement vector endpoints)
- however, so far was neglected what the role of kinetic visual stimuli may be. 

in the study: 
- dissected the collicular microcircuit responsible for the transfer of visual features, retina to collicular motor units. 
- characterised the responses to visual stims from premotor units in the SC
- characterised the conjunctive visual and motor tunig of identified individual units. 

finding: 
- SC motor units are tuned instead of to static features to kinetic features (visual flow opposite direction to head movement they decode).
- *network modell built on these alignment principles support key ethological functions in the sc such as prey capture*

conclusion: challenge the sensory-movement endpoint maping; favors a kinetic model based on alignment between sensory and movement vectors 

# Wilson et al. 2018

3d representation of the motor space in the mouse SC

note: the recording arrays are based on gyroscopes; head motion angular velocities are determined, and head motion events are detfined by a steady speed for a short period of time (100 ms (> 5 time bins) for over 0.5 degree per bin) (onsets and offsets are defined similarly). 
- Therein, it is here possible to measure general head motion events (full dynamic range, not limited by neck constraints - the mice can theoretically turn around). However, considering the task nature this is not the case (free foraging), as mice probably operate in a food-collecting strategy in which the "turn around" events are not needed at all. 
	-  this doesn't mean that the "around" space is not defined in the mice SC representation. 
	- it certainly is - especially on the sensory side. Let's refer to Masullo et al. 2019 for more details and a exposition. 
	- here I take note of what the present paper found.  

*Head displacements are unconstrained* (torsional component alive and well): 
- one can find the head displacement during free foraging be a bell-shaped curve around 0 degrees. The distribution widths are different across the RPY (roll has the smallest range, yaw and pitch not significanlty different.) In all these cases, rotations seldom exceed 90 degrees. 

RE- Donder's law (*oculootor control; angle of gaze is uniquely determined by the direction of gaze; not true in head rotations for mice*): although there is a reduced dynamic range for roll, it still is relevant; hence, the head rotation is not defined solely by the direction of gaze.  

(primates oculomotion adhere to this law; mice head rotations don't - roll is very relevant for a number of foraging tasks). 
- *constraints are laid out by the head-neck system*
For these constraints, the gimbal lock is not too much implicated (involved when say pitch is 90 degrees)

## tetrode bundles recording
- location: intermediate layers of the *left* SC. 
- functional: mice foraged a square open arena; light vs. dark conditions. 
- analysis: BTA (burst-triggered average) for each of the Euler components separately. (saccades are elicited by burst firing in the SC).
- of 300 neurons, about $65.1\%$ exhibit bursting activity, of which $16.1\%$ are consistently tuned to *angular head displacements around 1+ Eulerian components*. Of these tuned cells, $75\%$ are tuned to one component only (yaw ~ pitch > roll), the rest tuned to two components (mostly yaw x pitch and yaw x roll; no pitch x roll)
- notably, spikes outside of bursts would show less tuning than those occuring wihtin bursts. 

All cell are recorded in the left hemisphere
- for yaw: most cells are tuned toward the contralateral (clockwise, toward right) rotations ($83.3\%$)
- for pitch: most are tuned to downward motions ($76.5\%$) - this potentially has less thing to do to hemispheric assymetry as to the task itself. (or, very likely as well, to the natural task to the mice concerning head rotations - foraging!)
- for roll, all measured are tuned CCW rotations (reaching the mouth to the right)

- note only O(10) neurons are measured. 

## cells are found to be motion-tuned (as opposed to only direction tuned) - cell tuning are more variable between cells than within cell.
- *this is quite different from our network, if the interpretation is correct* - this suggests that the rotational space is partitioned not only by rotational axis but also by rotation amplitudes, across cells. 
- *in our case, cells seem to form a population code for rotational axis, and firing strength code for movement strength*.
	 - check "between-cell variability" vs. "within-cell variability" in our network. May this be artifact?
	 - literally, this reads to mean that BTA angles in these RPY are more stable between events of the same cell than across cells. 
		 - calculate the abs difference in mean displacement angle for each cell between the two light trials, and compare this to the abs difference between each cell and all other cells modulated to the same Eulerian comopnent
		 - record means in this metrics, perform analysis across all cells, and perform paired t-test


## firing rate vs. displacement/velocity

- in addition, the paper find the firing rate/firing duration do not correlate with motor displacement. What firing rates does correlate with is angular head velocity, in around $44\%$ of the cells. (*we can introduce velocity coding in our case too*)
	- **burst duration-velocity relationship?** [[question for Tripodi]]
- causality is not investigated (thus cannot know whether high FR causes high velocity, or high velocity -> sensory feedback -> high FR)

## visual cues and allocnetric heading

- recording in darkness: mainetained tuning for all Eulerian angles; accuracy of tuning is diminished wihtout visual cues/optic flow in some neurons. 

- for allocentric heading (external landmarks), there is no significant allocentric heading tuning in motion-tuned neurons; in some proportion, there is low-level modulation of firing rates. 
- takeaway: egocentric in nature, but sensory feedback from multiple modalities may be concerned - the loss of tuning in some cells here for visual feedback being an example.
- *intermediate layers of SC be a point of convergence of multiple sensory modalities (tactile, auditory and visual; bipartite or tripartite tuning)*
	- hence subpopulations may be influenced by different modalities, allowing sensory feedback integration that is robust and exhibit integration.
	
- relative role of each modality may be interesting
- earth vs. head centrism of SC (what are differences in the predictions?)
- topographic nature of motion vectors (anything similar to a retinotopic map for saccades, in head movements?)
	 - yaw and pitch are relevant here, and the involvement of roll is interesting. 
	 - head rotation for vision and head rotation for action may be different.


- cognitive: previous suggestion that the motor displacement vectors (recruited by the eye/head/reach systems) may serve to define a relational transformation map that determines object locations in the peripersonal space (motor-centric space encoding)
*it seems that our project is precisely on dissecting networks for generating spatially tuned head movements. we are also interested in the study of peripersonal space encoding generally.*


# Masullo et al. 2019

- *Pitx2* as a gene marker for a functionally homogneneous Glu population in the SGI (stratum griseum intermediate), which are tuned to specific head displacement vectors and optogenetic activation can trigger stereotypic head orienting movements with steplike kinematics.  
- these neurons *cluster in anatomically segregated modules that are the direct targets of patchy cortical afferents*
- functional modularity of these neurons seem to serve a spatial logic by being a convergence site for coherent sensory and motor signals of cortical and subcortical origins, instrumental in selection/execution of spatially oriented movments. 

## Identifying a cluster

step 1: use hierarchical clustering to identify functional diversity of neurons in the SGi - 5 functional classe, by patch-clamp recordings and in turn firing profiles

step 2: measure firing properties of excitatory/inhibitory clusters, by crosing Rosa-LSL-tdTomato reporter mice to vGluT2-CRE and vGAT-CRE mice (in which case, CRE is expressed in excitatory/inhibitory neurons, respectively). In this case, one can optogenetically manipulate activities of E/I cells.
- found that E cells populatd two classes of the similarity matrix (of firing profiles); I cells spread broadly. 

step 3: need further refinement in characterising the gene diversity within the SGI. 
- 3.1. screen gene expression in SC by extracting RNA from sensory/motor layer and sequenced the cDNA libraries. 
- 3.2. analyse sensory vs. motor gene expression for sites exclusive to the motor layer
- 3.3. found Pitx2 to be enriched in the motor area
	- these neurons account for about $23\%$ of all neurons in SGI, and share a unique electrophysiological profiles - (within one cluster out of the 5). They are all Gluergic (expression of vGluT2 RNA). 

## Tracing its connections

crossed Pitx2-CRE with Tau-LSL-FLPo-INLA mice then co-injected two FLP-dependent AAV viruses expressing a synaptic and a membrane marker in the SGI of these mice, to quantify downstream connetivity based on SynGFP intensity. 

- tectospinal tract connectivity (hence Pitx2On is a premotor population for the cephalomotor pathway)

## These neurons drive head displacements with stepwise kinematics

- put ChR2 in Pitx2ON SGI neurons (crossing Pitx2-CRE with Rosa-LSL-ChR2-eYFP mice) and implanted optic fibers above the SC. 
	- validate the system in slice preparation
	- test for causality by optogenetic stimulation of these neurons in free moving mice, while monitoring head/body kinematics. 

- Activation of these neurons in vivo --> robust and precise displacements of the head relative to body. 
- correlation between initial and final head-over-body positions is high (begin somewhere, end somewhere off to a linear offset) -> fixed motor program instead of fixed spatial position
	 - (though some SC neurons trigger flight responses via connections to dPAG; not here - so maybe distinct classes of SC neurons are involved)

stimulation for rates: not much modulation of the outcome displacements; stimulation duration however does affect the movment amplitude. There are "deflection points" for very long stimulations, represented by brief head movements opposite to the ongoing motion (reset command?)
 - for duration, these neurons seem to coordinate motions in "steps". 
 - ![[Pasted image 20240830111944.png]]
longer stim --> concatenated series of fixed characteristic movement vectors. 

## The above is exogenous stimulation; what about endogenous
- omitted details - basically they are active prior to head movements

## Motor map in the SC

### discrete modules for Pitx2ON neurons

![[Pasted image 20240830112833.png]]

### map of head movements by modules; 

- test if positional identity of the activated Pitx2-ON module determines the movement amplitudes/directions. 
- approach: place 9 optic fibers, measure average movement vectors. 

![[Pasted image 20240830114825.png]]

basically, each barrel seems to code for some stereotypic head movement.
 - what's weird is that the pitch dynamic range is not covered fully (only a very narrow subset!)

Perhaps in the present study we will find a few principles under which this type of motor map is generated, and we can thence put forward hypotheses about population coding

[[Dorrel et al 2023]]
