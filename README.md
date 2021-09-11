#### Heterogeneous Risk Minimization
> Jiashuo Liu

This repository contains the code for our ICML21 paper **Heterogeneous Risk Minimization**[1], including the implementation of HRM algorithm and 
the selection bias simulation data. 

Specifically, the repository contains the following files:
* `Selection_bias.py`: the implementation of our selection bias simulation data. Details of the functions included:
    * `data_generation`: the basic data generation function with respect to the equation (18) in the paper.
    * `modified_selection_bias`: when dealing with high dimensions of $V_b$ , the efficiency of original function 'data_generation' is quite low, and we propose a equivalent way to generate data.
    * `Multi_env_selection_bias` & `modified_Multi_env_selection_bias`: generate multi-environment training data. (The data are pooled together before inputting to the algorithm)
* `Frontend.py`: the implementation of the $\mathcal{M}_c$  model, which we implement as a clustering method 
* `Backend.py`: the implementation of the $\mathcal{M}_p$ model, which contains two parts: feature selection and invariant learning. Details of the classes included:
    * `FeatureSelector`: a feature selection module, for which we use the code from [2].
    * `MpModel`: the whole backend module.

Besides, there are many hyper-parameters to be tuned for the whole framework, which are different among different tasks and require users to carefully tune. During the experiments, we found serveral important factors and some intuitive tuning ways:

* `alpha`: this differs a lot among tasks, from 1e-1 to 1e3, and users may have to carefully tune it.
* `hard_sum`: in fact, this factor reflects the number of the ground-truth stable covariates. Since we have no idea the exact number of them, we propose to simply set it to the input number of covariates, and alternatively adjust the parameter `lam`.
* `Overall_threshold`: when the HRM algorithm gives the probabilities of covariates, we use a threshold to disgard the inferred unstable covariates by this threshold. As for tasks where the gaps of probabilities among different covariates are quite large, we simply disgard the covariates whose probabilities are below this(set to 0.20 in the simulation data). As for tasks where the gaps are small, we do not apply this and use the continuous probabilities in testing. 



Further, we view the proposed HRM as a general framework, which contains several techniques, including clustering, feature selection and invariant learning. Therefore, the components in our framework can be replaced by other methods. For example, in practice, the regularizer for invariant learning can be replaced by other invariant learning methods with multiple environments(though the theoretical properties might be affected...). And our proposed algorithm has many drawbacks:

* The convergence of the frontend module cannot be guaranteed, and we notice that there may be some cases the next iteration does not improve the current results or even hurts. 
* Hyper-parameters among different tasks are quite different.
* In this paper, we only conduct experiments under linear cases, and more complicated models are not tested yet(maybe later we will add...)



ps: I am really unsatisfied with the style of my code, and a better version is under development.  