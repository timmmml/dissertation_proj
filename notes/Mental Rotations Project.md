# Interim summary up until 24/09/2024

## Story line

- at the get-go, we were interested to find out about how a neural circuit can implement a rotation plan in 3D rotation. 
	- this is interesting because the group representation of the goal location is very different than that required by the motor system: you would need to infer a direction in every actual rotation you do!
	- finding: 
		- you can indeed train neural network structures to perform this rotations task: specifically, the performance is quite good when the problem is simple. generalisation to a harder problem is solely a network training problem, and can be solved with a conceivable transformation of the object to a simplified contour representation.  
	- resulting reps indeed capture rotation-specific characteristics (see plots in [[neurons]]). specifically, principle components use axis to represent rotational axis and the distance to the origin codes fo rthe rotation extent. we are able to (TODO - check this in exp-regulated networks) see that if we strike a balance constraining the network to perform the rotation fastly but with "upper caps"  on activity at any one time point, we would see this mapping near perfectly: a dramatic change will happen almost explicitly near 180 degrees. 
- we then moved to see if it's possible to do the same for a more general 3D image. In this case, we need a more powerful network. We tried ConvNeXtT achitectures pretrained on image classification, and although it works better with natural stims than artificial (cows > abstract shapes) the rotation inference seems not to be perfect. (note we should probaly fine tune the network on rotation classfiers.) neither does cow-specific VAE seem to work. 
	- we reasoned that this end-to-end approach might be difficult to train and generalisation may be poor anyways - humans rotate unseen objects all the time! 
	- so there could be other strategies at this: for example, we can imagine an autoencoder-type model that compresses some object into a very easily manipulatable form (such as some potato, or internal object classes such as cylinders, circles whatever), rotate that representation, and decode it into the original object
	- the same thing may be going on for motor planning actually: high-level abstraction of motor sequences are turned into detailed muscle activations: we sometimes model M1 as a dynamical system that freely evolves, but the dynamical system could be resulting some paired modules that implements the rotation: the state module containing actionable codes and feeds forward into an action module, which feeds back to the state module that predicts the next goal
		- the role of an upstream cortical area is to control the state module a state that it believes retains the goal. 
	
	- we hence designed a controller-predictor architecture to explicitly put this model trainable.We interpret the controller as a generic upstream network, whose input is the target rotation and whose outputs control a downstream actuator to perform the rotation. The catch is that it contains an internal model, in this case explicit, of the controlled network: in this case, the predictor. 
		- intelligent M1: control some output subnetwork such that it performs the output rotation via its internal dynamics. 
		- **or take this cross-regional**: some upstream area works on seeding M1 such that it performs the appropriate rotation: it utilises the feedback information from M1 and itself such (and only such) that it performs the prediction of the resulting rotation in its input space. 
		
- these two styles of modelling are going to be important in our next steps: 
	- investigate what's invariant across actuators, in (hierarchical vs. flat) * (controller-predictor vs. end-to-end) architectures
		- we will find out about decomposition of the computation in these architectures when we ask the network to divide and conquer the task in a way that it pleases - one that generalises on the output level across actuators, and on the input level across shapes. 
		- then we could also check out if natural manifolds exist, where re-aimg is simple or hard. 
	- the divide-and-conquer strategy can influence reaction time profiles aross times: 
		- vary object complexity: if there is a fixed "autoencoding" process, then the portion dedicated to the object complexity will be fixed per object. 
		- 


## Pretrain task 

Question: 
- given an analytical representation of rotation, what would an RNN learn when tasked to perform the target rotation?
- used several architectures: [[Meeting with Guillaume 0715]]
	- results: training was difficult for those cases where the quaternions are random.

- [[Interim Report]]