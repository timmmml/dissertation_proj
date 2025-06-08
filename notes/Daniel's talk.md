Suffciency: planning, observation, rewards

- this is about *belief states*
- two postulates for belief states: sufficiency, and markov conugacy 

if the belief state is good: 
- can predict future latents, observations, rewards, and values (under any policy) (for RL functions)
- (these translate to good losses to give!)

**(relationship to my proj: how to craft the state space)**

- concretize my case that in my problem the $\theta$ is a good stae

*RNN itself is a good inductive bias here! as compared to the no recurrence thing; the inductive bias is in that it introduces the filtering thing*
 (in my case we just push all states and hence the recurence within the CAN thing)






## how to make reps more belief like: 

1. learning a seond function from the hidden representation (auxiliary tasks in CV for example) *--> more belief-like states*


UNREAL (Jaderberg et al. 2016) and DreamerV3 for state discovery in the wild

UNREAL
$$
\begin{align}
L_{\text{UNREAL}} = \mathcal{L}_{A_{3}C} + \lambda _{VR} \mathcal{L}_{VR} + \lambda _{PC} \sum_{c}\mathcal{L}_{Q}^{(c)} + \lambda_{RP} \mathcal{L}_{RP}
\end{align}
$$
left to right: on policy, PC: auxiliary (pixel control policy) + RP: reward prediction
- predicting output giving action sequences is good too

Dreamer V3 (Hafner et al. 2025)

-  (discover diamonds in minecraft)

- architecture: world model + actor critic

WM: *calculate belief prior $p_{t}$ from previous belief $b_{t} + action$*
combine prior belief $p_{t}$ with likelihood from $o_{t}$ to sample the *latent posterior*

WM loss: 
$$
\begin{align}
\mathcal{L} = f(\beta \mathcal{L}_{\mathrm{p\mathrm{Re}d}} + \dots_{dyn} + \dots_{})
\end{align}
$$



Current talk: theoretically justify the good loss in the wild, form introducing belief states

