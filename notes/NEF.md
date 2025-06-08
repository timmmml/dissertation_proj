Overview: 
- a methodology to for *constructing simulations of neural systems*

principles: 

1. Neural reps: nonlinear encoding + weighted linear decoding
2. transformations of neural reps: functions neurally rep-ed variables; *transformations are determined using an alternately weighted linear decoding (transformational decoding)*
3. neural dynamics: neural reps as cnotrol-theoretic state variables. Analyze neural dynamics using control theory
4. must account for the effect of noise in each case

methodology: 

1. system description: to describe the neural system of interest such that the above principles are directly applicable: 
	1. use available neuroantomical data + functional understanding to *describe the architecture, functions, and representations*; interconnectivity between subsystems, neuron response functions, tuning curves, subsystem functional relations, and overall system behavior
	2. write out everything in maths, such that mathematical transformations are defined to describe the "functions" specified. *translate the functional description, in neurobiology, to that in mathematics*
	
2. design specification: real-world limitations to be assumed for the neural system. "implementational constraints". Determine the appropriate kind of representation for each variable, define dnamic ranges, precision, and signal-noise ratio for each DoF. 
	1. power spectrum/bandwidth must be specified, made on the basis of available data. 
	2. define the real-world operating constraints such that the system can successfully perform the function. 

3. implementation: combine the above steps to determine appropriate encoding and decoding rules to implement the specified behavior. 
	1. encoding: higher-order representations --> lower-order representations
	2. transformation of rep at a level of representation: compute linear decoding that meets design specs/connection weights between neurons
	- we determine which aspects of the system would be simulated with great/lesser details. (to determine when to use conductances and when to consider neural populations)

