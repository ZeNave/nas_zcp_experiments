# nas_zcp_experiments
Experiments were done to discover how much information there is in the Jacobian matrix of a neural network at initialization.


This code should be used in conjunction with the code and auxiliary files from [https://github.comBayesWatch/nas-without-training](https://github.com/BayesWatch/nas-without-training) and from [www.github.com/VascoLopes/EPENAS](https://github.com/VascoLopes/EPE-NAS). You should follow their instructions on how to set up your environment.

First, you should run create_traindataset.sh to create your dataset, which associates each Jacobian matrix batch to that architecture's corresponding performance.

Secondly, you should run the train.sh file to train a chosen model in the previously created dataset.

Lastly, you should run the my_reproduce.sh file to use the trained model on the same conditions as swot and depends.
