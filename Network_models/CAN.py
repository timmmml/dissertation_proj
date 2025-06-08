"""This module implements the continuous attractor network model. 

- the dissertation is about how a continuous attractor network can be arrived. For doing that, we are to get the network to take in group Lie algebras and train CANs based solely on input self-consistency
- therefore, on the model definition part, we only care about a dynamical structure. 
- start with RNN then try other recurrent nets.


One key difficulty is to deal with periodicity in the input. Maybe it doesn't matter - we don't need periodic tuning curves, as we only care about moderately sized inputs on any time point. Instead, we care about how the network integrates these inputs.  
"""

# sounds easy enough. let me try to do it in a notebook first. Coming back in a bit.