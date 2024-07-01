# Day 1: 01/07/2024

## Logs 

Goal: 
- set up basic training skeleton on pycharm.
	- [x] Set up the virtual environment to put in pycharm initially
	- [ ] Set up basic training skeleton
		- [ ] Deployment within `Network_models` and `Training`
          - In `Network_models`:
            - [ ] Base network_model object which extends from the `torch.nn.Module` class
            - [ ] Create a simple RNN model that can be configured (input and output settings)
          - In `Training`:
			- [ ] Base trainer object (configure as a handle which accepts directly a configuration and outputs training 
			  logs as well as stores the trained model.)
- brush up on group theory and SO(3): Lie group, Lie algebras, and the exponential map. Temporary aim is to know enough to implement stage 1 training of the model (specified rotation either via a rotation matrix (SO(3) member) or a quaternion (unit quaternion)). 
	- [x] Lie group basics
	- [x] Lie algebras basics
	- [x] exponential map basics
	- [ ] [[Quaternion integration]]
	- [ ] [[Quaternion distance metrics]]

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
  