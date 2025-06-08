Survey on zeroing the null representation
- wouldn't work if both get zero rep and zero bias 
- in that case, there is no dynamics considering our architectural design 
$$
\begin{align}
g_{t + 1}^{\text{rep}} &= \alpha(W_{rr} g_t^{\text{rep}} + \sum_{j \in [1, \mathrm{J}]}u_t^jW_{ir}^{j} g_t^{\text{in}, j} + b_r)\notag\\
g_{t + 1}^{\text{in}, j} &= \alpha(W_{ri}^{j} g_t^{\text{rep}} + \sum_{j \in [1, \mathrm{J}]}u_t^jW_{ii}^{j} g_t^{\text{in}, j} + b_i^{j})\notag
\end{align}
$$
However, once one allows bias input (or bias in rep), this thing becomes at least trainable
[[but the "zero rep" is not actually the null rep]]
(this is because somehow the network can learn to drift toward a non-zero null; it is also hard for some uniform state to be stable)
also, if it is, then pattern formation around it is a bit tricky.

but maybe...? (I will double check but let's shelf this for now)
- I fixed the training loop to correctly encourage loss function on the initial rep, such that the drift situation is penalized appropriately
- in the loss term, average dissimilarity matrices for across the equivalence class (positive pair, evaluated at stimulus end, signal end (after zero-padding), and a random position)

one situation that seems to be close to what we want: 
- the manifold at zero centers around the zero initial condition; 
![[Pasted image 20250129131636.png]]
evolves around zero (red = trial start; blue = trial end)
![[Pasted image 20250129131813.png]]
On a 2D torus: 
![[Pasted image 20250129132314.png]]
![[Pasted image 20250129132354.png]]

Anyways, shelf this for now. 

--- 

![[controller design]]
