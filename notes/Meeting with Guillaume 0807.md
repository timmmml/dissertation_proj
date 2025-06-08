## Progress catchup:
- forward model now ready to train
	- found an adequately performing set of hyperparameters to train the forward model: (see J. notebook)
	- the best performing model with quaternions used ~350000 datapoints to train, each consisting of 5 snapshots of quaternions. 
	- performance is not bad: the final geodesic loss is around 0.065 on a validation set - (I didn't bother to plot boxplots for this, just a proof of concept that this type of modelling can work. The model seemed to have converged in 630 epochs, each of 8192 samples and minibatch size of )
	- this will lead to shocking storage requirements if we store training data in images.

- Hence, would investigate various types of pre-trained encoding  models to drastically downsize the data. 

- An example of trained results can be seen. 