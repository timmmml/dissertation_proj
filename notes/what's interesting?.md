## my "dream" project: 

- pursue brain representation of any motor plan, such that it is feasible to decode any motor plan from neural activity. 
- example goal 1: a flexible device that the brain can learn to control and form well-integrated, free motor plans. 
- example goal 2: 
	want to study: core representations of actions (dynamics underlying motor preparation),
	which feeds into the engineering of: unsupervised learning for motor plan decoding; some online filter mechanism that allows real-time intention-decoding from neural activity (and something like brain-copilot which is integrated within an AR setup)
	- or very general exoskeleton device that replaces the (lenovo trackpoint keyboard + vim) setup. this can with the help of copilots create super programmers! *I will have the computer do anything I imagine*
		- this is quite good actually as a final goal (not likely done before I have my PhD at least). Cuts into decoding a very general set of motor plan I do with my hands that are better defined by underlying "action motifs" than by hand movements
			- language production (typing)
			- moving on my document (vim)
			- manipulating 3D objects (by various mouse actions etc.)

## my quest for "interesting" stuff

(mental rotations as a starting point)
- what's interesting would be that the target rotation is on SO(3) but the controller represents the motor plan as a chosen way to achieve the target rotation
	- at some point in the brain this must happen. where? (closer to the motor end, vs. closer to the sensory end)
	- can investigate this by looking at hierarchical predictive models! i.e., the motor layer is fed back to sensory layers to allow it predict the updated sensory state. The sensory state would originally only depend on the SO(3) nature, but it *must* also deal with efference copies from the motor side. *what's the most efficient way this can be solved by a neural system?*
		- may be the case that there exists a hierarchy of population codes for the motor plan (one set of cells code for the target rotation without the direction, the other code for the direction as well.)
		- you will need the trajectory of the head to meaningfully integrate visual feedback en route the rotation; you will also need to predict the sensory state after the rotation is done. 
		- would also be important to consider speeds of rotation (SC neuron FR controls speed not displacement)

- the experiment (in the best world where I get to design this) would be a *rotate to target* system, in which we design a task that involves some target rotation, sensory feedback, and a motor plan. 

- mice: design a "3D" environment where the mice can somehow manipulate the environment to get the target rotation (this will be a kind of expensive task that won't be doable within my time frame)
- human: 
1. VR task where the human must manupulate a virtual hand or something 
2. the classical mental rotation task. however, the original task cannot solve what we want to solve (from the classical behavioural metrics we cannot infer whether the motor plan has a direction and how it interacts with feedback)
	- maybe can use some actual online feedback to see here. 
	- entertain the idea of rollouts
	- need to read more on motor control literature to figure out how feedback is integrated and how it manifests in behaviours

other stuff that warrants attention: in the case of human (monkey) manual rotation of something, stuff that could be interesting are:
- to enforce transformations in the 3D space, humans must control the same effector system for different tasks: for example, when building an axe, one must both bring the components together and rotate them to fit (at the same time?)
- one can have a general system (for example SE(3)) whose transformation that neurons can code for, or have two subsets of neurons that code for these transformations separately. 
- (waht about geometric algebra?)