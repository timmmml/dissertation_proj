1D torus: 
- p = 10, bReLU, normalize
 ![[Pasted image 20241204132959.png]]
- p = 1, bReLU, normalize
![[Pasted image 20241204133117.png]]
but if we half the invariance constant
![[Pasted image 20241204133537.png]]
doubling it: 
![[Pasted image 20241204133704.png]]

2D torus
example: use capacity loss, normalize
invariance weight = 1
![[Pasted image 20241204134225.png]]
invariance weight = 0.5
![[Pasted image 20241204134352.png]]

Example 1D control cells: 
![[Pasted image 20250102095241.png]]

Example 2D control cells: (the selection width could be different in each cell, but correlated across the from- and to- angle).
![[Pasted image 20250102095125.png]]
![[Pasted image 20250102095149.png]]
![[Pasted image 20250102095204.png]]

if minimize control norm, we get more or less similar stuff
![[Pasted image 20250102100234.png]]
this one is clearly doubly tuned: 
![[Pasted image 20250102100333.png]]
