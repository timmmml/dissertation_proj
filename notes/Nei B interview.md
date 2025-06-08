Motivation: to understand the hierarchical information in the brain with computational principles, treating it as an adaptive robot learning algorithm. 
- think about the sensory loop: the brain learns an efficient way understand the latent causes of sensory observations for behavioral purposes &c. 
- in the motor control as well, there is a hierarchy that comes from decision making to motor outputs. This comes at different levels of abstraction. 
	- if we think about goal directed behavior, an action plan in some goal domain is transformed into an action plan in the muscle space. 

It is very important to understand the factorization of behavioral control as well as how this factorization is achieved in the brain, in a way that hopefully facilitates optimality principles such as for transfer learning. 

this fits greatly with Tim Behrens's lab whose goal is almost precisely to uncover how complex behavior is structured, how new plans are composed (say compositional hippocampal replay), and neural representations. 
his and your work in hippocampus is fundamental in this issue, because I think hippocampus is kind of forced to be efficient in its representation of the world such that new behaviors can be learned and the action repertoire to be enriched. I am currently thinking about hierarchical motor control and the cortex, but I am very eager to explore much more about the hippocampus and hierarchical plannig (juxtaposed to that of motor control, where I think the hierarchy lies in the parametrization of the action and derives from structural connections across the cortex)

My current project: 
- first half: descriptive tool for going from a manifold description of the state space to a continuous attractor network. (assume a neural population holds some state space correctly, what shall it look like?)
	- follow the postulates 
	- (question for Neil: your lab does CAN work in a very recent paper which I am only beginning to unpack)
	- training with minimal assumptions + appropriate bio constraints

- end product: a network that integrates your actions in a way that respects similarity principles in the manifold. 

Lent: 
 - investigate how we attach meaning to this manifold in control, by co-learning a sensory-prediction network and a state-space controller on top of this CAN. 
 - now, it's one thing to use a pretrained attractor; it's another thing to train this stuff from end-to-end. In my past summer's project which revolves around a controller in SO(3), the learned result is a half-space that correctly respects the minimal energy control (introduce break-off points in the goal state space; not SO(3) but something closer to a half of S^3 (where we constrain the real component to be positive)) 
- also the sensory feed-in becomes a parametrization of the state space in itself. 
	- this is not necessarily good if we want to generalize the structural knowledge of the problem! (think about multiple objects to manipulate --> shared representation and control --> mulitple effector). The most efficient way to transfer-learn is to preserve some kind of control over the shared manifold, and learn things in separation as mapping problems instead of learning everything end-to-end, always


Question: 
- what are 