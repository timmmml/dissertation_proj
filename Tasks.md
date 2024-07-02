# Pretraining
## Task 0

| Task id | Task Specification                              | Task input                                         | Task output                                           |
| ------- | ----------------------------------------------- | -------------------------------------------------- | ----------------------------------------------------- |
| 0.1q    | pre-train task 1q: copy the rotation (quat)     | 500 steps of quaternion, 500 steps of silence      | 500 steps of silence, 500 steps of angular velocities |
| 0.1m    | pre-train task 1m: copy the rotation (matrix)   | 500 steps of a rotation  matrix, 500 steps silence | 500 steps 0, 500 steps $\hat{\omega}$                 |
| 0.2q    | pre-train task 2q: invert the rotation (quat)   | 500 ms $q^{*}$,  500 ms $0$                        | 500 ms $0$, 500 ms $\hat{\omega}$                     |
| 0.2m    | pre-train task 2m: invert the rotation (matrix) | 500 ms $R^{T}$, 500 ms $0$                         | 500 ms $0$, 500 ms $\hat{\omega}$                     |
Try to implement Euler angles here too. 