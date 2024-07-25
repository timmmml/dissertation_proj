# Branch intro: 

This branch is a lean version of the project. 

- all notes are removed (they are there on the branch called tims, which will be used to carry around stuff on system reboots &c. (recent switch to Linux taught me the lesson of making everything easily restorable from Github.)). So are .obsidian/.idea &c. Those are settings of cojunct programs I use alongside the project, and carry no meaning for the actual project. GUI is removed. All visualisation will be local
- further, I will consider removing jupyter notebooks used for exploratory training and visualisation, and switch to a full .py-based operations stream. Only one notebook will be left, as a master console that calls these subroutines. 

# Division of labor: 

on_nonnormal is used to store all files relevant for large-scale model training. 

This includes: 
- model dataset generation (rendering of images & c.)
- model training
- model evaluation (logs and streamlined visualisations)

tims is used to explore and analyse model performance.
- once models are trained, they are streamed back to the local machine for evaluation and can be accessed from there. 