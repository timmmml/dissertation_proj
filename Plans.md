# Neural Representation of 3D Rotations in Superior Colliculus

## Project Overview

This project aims to investigate how neural circuits represent 3D rotations, focusing on the superior colliculus in mice. The study will combine computational modeling with behavioral experiments to understand the neural basis of mental rotation tasks.

## Objectives

1. Develop a CNN-RNN model capable of representing and executing 3D rotations
2. Generate and test hypotheses about neural representations in the superior colliculus
3. Collect and analyze behavioral data on 3D mental rotation tasks
4. Compare model predictions with neural and behavioral data

## Methodology

### 1. Stimulus Generation

- Use OpenGL to create image pairs representing objects in different 3D orientations
- Develop a dataset of stimuli covering a range of rotation angles and axes

### 2. Model Architecture

- Design a CNN-RNN structure
  - CNN: for processing input images
  - RNN: for digesting and unfolding rotation information over time

### 3. Model Training

#### Phase 1: Pre-training

- Train the model to execute 3D rotations using infinitesimal moves along rotational axes
- Implement a supervision signal based on rotation actions over time
- Ensure the model can maintain rotation information in the absence of input

#### Phase 2: Task Training

- Adapt the Shepard & Metzler mental rotation task for arbitrary 3D rotations
- Train the model to perform the mental rotation task

### 4. Behavioral Experiments

- Design and conduct experiments to measure human performance on 3D mental rotation tasks
- Focus on collecting reaction time data

## Further steps...
### 5. Superior Colliculus Data Analysis

- Analyze neural recordings from mice superior colliculus during rotation tasks
- Compare neural activity patterns with model predictions

### 6. Hypothesis Generation and Testing

- Formulate hypotheses about neural representations of 3D rotations based on model insights
- Test these hypotheses using the superior colliculus data

## Timeline

1. Week 1: Literature review (Group representation theory, Attractor Dynamics SO(3), m-GPLVM), project directory setup [[Week 1]]
2. Week 2-3: Stimulus Generation and model architecture development
3. Week 4: Model pre-training
4. Weeks 5-6: Representation analysis. 
1. Weeks 7-8: Programming the behavioural task interface (may use Gorilla? It's a 2AFC task with image inputs) + data collection. Task-specific model training.
2. Weeks 9-10: data analysis and report writing.

3. Further 1: Superior colliculus data analysis
4. Further 2: Hypothesis testing and result interpretation

## Expected Outcomes

1. A computational model capable of representing and executing 3D rotations
2. Insights into the neural mechanisms underlying 3D mental rotations
3. Comparison of model predictions with behavioral and neural data
4. Potential applications in understanding spatial cognition and motor planning

## Future Directions

- Extend the model to other brain regions involved in spatial cognition
- Investigate the development of 3D rotation representations during learning
- Apply findings to improve brain-computer interfaces for 3D object manipulation

