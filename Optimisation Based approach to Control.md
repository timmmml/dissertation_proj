**Exam**
The aims of the course are to:

- introduce methods for feedback system design based on the optimization of an objective, including reinforcement learning and predictive control.
- demonstrate how such control laws can be computed and implemented in practice.

## Objectives

As specific objectives, by the end of the course students should be able to:

- understand the derivation and application of optimal control methods.
- appreciate the main ideas, applications and techniques of predictive control and reinforcement learning.

## Content

#### Optimal Control (7L + 1 examples class, Prof F Forni)

- Formulation of convex optimisation problems
- Status of theoretical results and algorithms
- Formulation of optimal control problems. Typical applications
- Optimal control with full information (dynamic programming)
- Control of Linear Systems with a quadratic objective function
- Output feedback: ‘LQG’ control
- Control design with an “H-infinity” criterion

#### Predictive Control and an Introduction to Reinforcement Learning (7L + 1 examples class, Prof G Vinnicombe)

- What is predictive control? Importance of constraints. Flexibility of specifications. Typical applications
- Basic formulation of predictive control problem without constraints and the receding horizon concept. Comparison with unconstrained Linear Quadratic Regulator
- Including constraints in the problem formulation. Constrained convex optimization
- Terminal conditions for stability 
- Emerging applications: advantages and challenges
- Policy and generalized policy iteration; rollout algorithms and predictive control
- Approximate dynamic programming
- Deep neural nets as universal approximators for value and policy.
- Simulation based vs state space models - Q learning.