# Logs

Formatting: 
- [ ] = To do
- [?] = In progress (partially completed)
- [x] = Completed

## Day 1: 01/07/2024

Goal: 
- set up basic training skeleton on pycharm.
	- [x] Set up the virtual environment to put in pycharm initially
	- [?] Set up basic training skeleton
		- [?] Deployment within `Network_models` and `Training`
          - In `Network_models`:
            - [?] Base network_model object which extends from the `torch.nn.Module` class
          - In `Training`:
			- [?] Base trainer object (configure as a handle which accepts directly a configuration and outputs training logs as well as stores the trained model.)
	- [x] Familiarise with quaternion operations in Python
		- [x] scipy.spatial.transform.Rotation implementation class for straightforward operations.
		- [x] implementation of the Geodesic distance computation between 
			- [x] two quaternions
			- [x] two rotational matrices
			- [x] two Euler angles (by transforming to quaternions)
		- [x] implement basic quaternion operations in torch.tensor for autograd computation (quaternion multiplications, exponential map)
	
- brush up on group theory and SO(3): Lie group, Lie algebras, and the exponential map. Temporary aim is to know enough to implement stage 1 training of the model (specified rotation either via a rotation matrix (SO(3) member) or a quaternion (unit quaternion)). 
	- [x] Lie group basics
	- [x] Lie algebras basics
	- [x] exponential map basics
	- [x] Quaternion integration
	- [x] Quaternion distance metrics

## Day 2: 02/07/2024

- Monday's	work continued
	- [ ] Set up basic training skeleton
		- [ ] Deployment within `Network_models` and `Training`
          - In `Network_models`:
            - [ ] Base network_model object which extends from the `torch.nn.Module` class
            - [ ] Create a simple RNN model that can be configured (input and output settings)
          - In `Training`:
			- [ ] Base trainer object (configure as a handle which accepts directly a configuration and outputs training 
			  logs as well as stores the trained model.)

- Meeting with Guillaume (1600-1700)

# Notes

## Notes on model designed

README, step 1
- Model input types: 
  - [ ] Rotation matrices (3x3)
  - [ ] Euler angles (3)
  - [ ] Quaternions (4)
	  - [ ] continuous preparation (sustained input for 500 ms)
      - [ ] instantaneous preparation (input at time of movement)
- Model output types: 
  - [ ] Rotation matrices (3x3)
  - [ ] Euler angles (3)
  - [ ] Quaternions (4)
    - [ ] output angular velocities for integration (exponential map)
    - [ ] output modular, discrete rotations per step. 
  