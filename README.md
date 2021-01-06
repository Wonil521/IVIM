# Quantification of Intravoxel Incoherent Motion with Optimized b-values using Deep Neural Network
Pytorch implementation of IVIM quantification with optimized b-values using deep neural network. 

## Usage
For data generation, \
<code> python data_gen.py --num_bvalue [number of b-values to optimize, (ex) 4] --SNR [Target SNR, (ex) 100] --data_path [data path, (ex) '/home/user/'] </code> \
\
For optimizing b-values & training deep neural network, \
<code> python train.py --num_bvalue [number of b-values] --SNR [target SNR] --num_trials [number of trials] --gpu [gpu number] --lambda1 [lambda1, weighting of f] --lambda2 [lambda2, wieghting of D] --lambda3 [lambda3, weighting of Dp] --data_path [data_path] </code> \
\
For testing the trained deep neural network & optimized b-values\
<code> python test.py --test_num_bvalue [number of b-values] --SNR [target SNR] --test_trials [number of trials] --gpu [gpu number] --lambda1 [lambda1, weighting of f] --lambda2 [lambda2, wieghting of D] --lambda3 [lambda3, weighting of Dp] --data_path [data_path] </code>\
\
The default setting: number of b-values = 4, SNR=100, data path = 'home/user/', number of trials for training = 20, number of trials for test = 50, gpu = 0, lambda1 = 1.0, lambda2 = 1.0, lambda3 = 1.0.

# Optimized b-values
![github](https://user-images.githubusercontent.com/59683100/103767261-ab3fc680-5063-11eb-83c9-e601e15ea3d1.png)
