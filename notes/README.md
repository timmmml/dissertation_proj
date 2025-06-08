# Mental rotations UROP project

Mental rotations in RNNs

1. Train an RNN to take as input _some_ representation of a desired rotation during preparation, and to subsequently produce that rotation in an _autonomous_ fashion (i.e. with no input anymore).
   The following recipe is agnostic to the way the desired rotation is fed into the network ("_some_"). See points 2 - 8 for specific variations.
  a. Define your model: e.g. GRU with $N$ neurons. Linear 3D readout into angular velocity.
  b. [Quaternion integration](https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/) of that output during the entirety of the movement phase (say 500 ms) yields what we take to be the "rotational output" of the network. 
  c. Define a sensible loss function based on rotational distance (pick a sensible `SO(3) x SO(3) -> R` distance function; needs to be differentiable).
  d. Train the network by minimizing this loss function, regularized by the squared norm of the momentary outputs, averaged over trials. Each trial is defined by a different desired rotation.

2. Input a unit quaternion representation of the desired rotation (4 scalar inputs channels). Train.
3. Big step up: input a conv-net-massaged pixel-based representation of the (initially rotated) object to be rotated back into a fixed view (always the same across trials). 

```ocaml
let _resign (a, u) = if a > 0. then a, u else -.a, Mat.neg u

let create theta u =
  (* make sure u is normalised *)
  let u = Mat.(u /$ l2norm' u) in
  let c = Maths.cos (theta /. 2.) in
  let s = Maths.sin (theta /. 2.) in
  (c, Mat.(s $* u)) |> _resign
```

4. Investigate pre-training a "forward model" whereby the network learns the visual consequences of outputting a particular sequence of infinitesimal rotations.
5. Investigate the use of planning by covert rollouts.
   
