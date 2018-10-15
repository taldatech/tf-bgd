![title](https://github.com/taldatech/tf-bgd/blob/master/imgs/bgd_logo.png)
![subtitle](https://github.com/taldatech/tf-bgd/blob/master/imgs/bgd_subtitle.png)
# tf-bgd
## Bayesian Gradient Descent Algorithm Model for TensorFlow
![regress](https://github.com/taldatech/tf-bgd/blob/master/imgs/line.gif)

Python and Tensorflow implementation of the Bayesian Gradient Descent algorithm and model

### Based on the paper "Bayesian Gradient Descent: Online Variational Bayes Learning with Increased Robustness to Catastrophic Forgetting and Weight Pruning" by Chen Zeno, Itay Golan, Elad Hoffer, Daniel Soudry

Paper PDF: https://arxiv.org/abs/1803.10123

## Theoretical Background

The basic assumption is that in each step, the previous posterior distribution is used as the new prior distribution and that the parametric distribution is approximately a Diagonal Gaussian, that is, all the parameters of the weight vector $\theta$ are independent.

We define the following:
* ![equation](https://latex.codecogs.com/gif.latex?%24%5Cepsilon_i%24) - a Random Variable (RV) sampled from ![equation](https://latex.codecogs.com/gif.latex?%24N%280%2C1%29%24)
* ![equation](https://latex.codecogs.com/gif.latex?%24%5Ctheta%24) - the weights which we wish to find their posterior distribution
* ![equation]($\phi = (\mu,\sigma)$) - the parameters which serve as a condition for the distribution of ![equation](https://latex.codecogs.com/gif.latex?%24%5Ctheta%24)
* ![equation](https://latex.codecogs.com/gif.latex?%24%5Cmu%24) - the mean of the weights' distribution, initially sampled from ![equation](https://latex.codecogs.com/gif.latex?N%280%2C%5Cfrac%7B2%7D%7Bn_%7Binput%7D%20&plus;%20n_%7Boutput%7D%7D%29)
* ![equation](https://latex.codecogs.com/gif.latex?%5Csigma) - the STD (Variance's root) of the weights' distribution, initially set to a small constant.
* ![equation](https://latex.codecogs.com/gif.latex?K) - the number of sub-networks
* ![equation](https://latex.codecogs.com/gif.latex?%5Ceta) - hyper-parameter to compenstate for the accumulated error (tunable).
* ![equation](https://latex.codecogs.com/gif.latex?L%28%5Ctheta%29) - Loss function

Algorithm Sketch:

* Initialize: ![equation](https://latex.codecogs.com/gif.latex?%5Cmu%2C%20%5Csigma%2C%20%5Ceta%2C%20K)
* For each sub-network k: sample ![equation](https://latex.codecogs.com/gif.latex?%5Cepsilon_0%5Ek) and set ![equation](https://latex.codecogs.com/gif.latex?%5Ctheta_0%5Ek%20%3D%20%5Cmu_0%20&plus;%20%5Cepsilon_0%5Ek%20%5Csigma_0)
* Repeat:

    1. For each sub-network k: sample ![equation](https://latex.codecogs.com/gif.latex?%5Cepsilon_i%5Ek), compute gradients: ![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20L%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta_i%7D)
    2. Set ![equation](https://latex.codecogs.com/gif.latex?%5Cmu_i%20%5Cleftarrow%20%5Cmu_i%20-%20%5Ceta%5Csigma_i%5E2%5Cmathbb%7BE%7D_%7B%5Cepsilon%7D%5B%5Cfrac%7B%5Cpartial%20L%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta_i%7D%5D)
    3. Set ![equation](https://latex.codecogs.com/gif.latex?%5Csigma_i%20%5Cleftarrow%20%5Csigma_i%5Csqrt%7B1%20&plus;%20%28%5Cfrac%7B1%7D%7B2%7D%20%5Csigma_i%5Cmathbb%7BE%7D_%7B%5Cepsilon%7D%5B%5Cfrac%7B%5Cpartial%20L%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta_i%7D%5Cepsilon_i%5D%29%5E2%7D%20-%20%5Cfrac%7B1%7D%7B2%7D%5Csigma_i%5E2%5Cmathbb%7BE%7D_%7B%5Cepsilon%7D%5B%5Cfrac%7B%5Cpartial%20L%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta_i%7D%5Cepsilon_i%5D)
    4. Set ![equation](https://latex.codecogs.com/gif.latex?%5Ctheta_i%5Ek%20%3D%20%5Cmu_i%20&plus;%20%5Cepsilon_i%5Ek%20%5Csigma_i) for each k (sub-network)
    
* Until convergence criterion is met
* Note: i is the ![equation](https://latex.codecogs.com/gif.latex?i%5E%7Bth%7D) component of the vector, that is, if we have n paramaters (weights, bias) for each sub-network, then for each parameter we have ![equation](https://latex.codecogs.com/gif.latex?%5Cmu_i) and ![equation](https://latex.codecogs.com/gif.latex?%5Csigma_i)

The expectactions are estimated using Monte Carlo method:

![equation](https://latex.codecogs.com/gif.latex?%5Cmathbb%7BE%7D_%7B%5Cepsilon%7D%5B%5Cfrac%7B%5Cpartial%20L%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta_i%7D%5D%20%5Capprox%20%5Cfrac%7B1%7D%7BK%7D%5Csum_%7Bk%3D1%7D%5E%7BK%7D%5Cfrac%7B%5Cpartial%20L%28%5Ctheta%5E%7B%28k%29%7D%29%7D%7B%5Cpartial%20%5Ctheta_i%7D)


![equation](https://latex.codecogs.com/gif.latex?%5Cmathbb%7BE%7D_%7B%5Cepsilon%7D%5B%5Cfrac%7B%5Cpartial%20L%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta_i%7D%5Cepsilon_i%5D%20%5Capprox%20%5Cfrac%7B1%7D%7BK%7D%5Csum_%7Bk%3D1%7D%5E%7BK%7D%5Cfrac%7B%5Cpartial%20L%28%5Ctheta%5E%7B%28k%29%7D%29%7D%7B%5Cpartial%20%5Ctheta_i%7D%5Cepsilon_i%5E%7B%28k%29%7D)

### Loss Function Derivation for Regression Problems

![equation](https://latex.codecogs.com/gif.latex?L%28%5Ctheta%29%20%3D%20-log%28P%28D%7C%5Ctheta%29%29%20%3D%20-log%28%5Cprod_%7Bi%3D1%7D%5E%7BM%7D%20P%28D_i%7C%5Ctheta%29%29%20%3D%20-%5Csum_%7Bi%3D1%7D%5E%7BM%7D%20log%28P%28D_i%7C%5Ctheta%29%29)

Recall that from our Gaussian noise assumption, we dervied that the target (label) ![equation](https://latex.codecogs.com/gif.latex?t) is also Gaussian distributed, such that: ![equation](https://latex.codecogs.com/gif.latex?P%28t%7Cx%2C%5Ctheta%29%20%3D%20N%28t%7Cy%28x%2C%5Ctheta%29%2C%20%5Cbeta%5E%7B-1%7D%29)
where ![equation](https://latex.codecogs.com/gif.latex?%5Cbeta) is the percision (the inverse variance).
Assuming that the dataset is IID, we get the following:
![equation](https://latex.codecogs.com/gif.latex?P%28t%7Cx%2C%5Ctheta%2C%20%5Cbeta%29%20%3D%20%5Cprod_%7Bi%3D1%7D%5E%7BM%7D%20P%28t_i%7Cx_i%2C%5Ctheta%2C%20%5Cbeta%29)
Taking the negative logarithm, we get:
![equation](https://latex.codecogs.com/gif.latex?-log%28%20P%28t%7Cx%2C%5Ctheta%2C%20%5Cbeta%29%29%20%3D%20%5Cfrac%7B%5Cbeta%7D%7B2%7D%5Csum_%7Bi%3D1%7D%5EM%20%5By%28x_i%2C%5Ctheta%29%20-%20t_i%5D%5E2%20-%5Cfrac%7BN%7D%7B2%7Dln%28%5Cbeta%29%20&plus;%20%5Cfrac%7BN%7D%7B2%7Dln%282%5Cpi%29)
Maximizing the log-likelihood is equivalent to minimizing the sum: ![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bi%3D1%7D%5EM%20%5By%28x_i%2C%5Ctheta%29%20-%20t_i%5D%5E2) with respect to ![equation](https://latex.codecogs.com/gif.latex?%5Ctheta%24) (looks similar to MSE, without the normalization), which is why we use `reduce_sum` in the code and not `reduce_mean`.

Note: we denote D as a general expression for the data, and in our case is the probability of the target conditiond on the input and the weights. Pay attention that ![equation](https://latex.codecogs.com/gif.latex?L%28%5Ctheta%29) is the log of the probability which is log of an expression between [0,1], thus, the loss itself is not bounded. The probability is a Gaussian (which is of course, bounded).

### Regression using BGD

We wish to test the algorithm by learning ![equation](https://latex.codecogs.com/gif.latex?y%20%3D%20x%5E3) with samples from ![equation](https://latex.codecogs.com/gif.latex?y%20%3D%20x%5E3%20&plus;%5Czeta) such that ![equation](https://latex.codecogs.com/gif.latex?%5Czeta)~![equation](https://latex.codecogs.com/gif.latex?N%280%2C9%29). We'll take 20 training examples and perform 40 epochs.

#### Network Prameters:
* Sub-Networks (K) = 10
* Hidden Layers (per Sub-Network): 1
* Neurons per Layer: 100
* Loss: SSE (Sum of Square Error)
* Optimizer: BGD (weights are updated using BGD, unbiased Monte-Carlo gradients)



## Prerequisites
|Library         | Version |
|----------------------|----|
|`Python`|  `3.6.6 (Anaconda)`|
|`tensorflow`|  `1.10.0`|
|`word2keypress`|  `-`|
|`numpy`|  `1.14.5`|

## Basic Usage

## Files in the repository

|File name         | Purpsoe |
|----------------------|----|
|`trans_dict_2idx.json`|  Transformation to an index.|


## Main Example App:

You should only use the `pass2path_v2.py` file with the following arguments:

|Argument                 | Description                                 |
|-------------------------|---------------------------------------------|
|-h, --help       | shows arguments description             |
|-t, --train      | train pass2path model                   |
|-p, --predict    | predict using a trained pass2path model |
|-x, --test       | test (file) using a trained pass2path model |
|-q, --residual| use residual connections between layers (training) |
|-o, --save_pred| save predictions to a file when testing (testing) |
|-d, --dataset| path to a `.csv` dataset (training, testing)|
|-c, --cell_type| RNN cell type - `lstm` or `gru` (training, default: `lstm`) |
|-s, --step| display step to show progress (training, default: 100) |
|-e, --epochs| number of epochs to train the model (training, default: 80) |
|-b, --batch_size| batch size to draw from the provided dataset (training, testing, default: 50) |
|-z, --size| rnn size - number of neurons per layer (training, default: 128) |
|-l, --layers| number of layers in the network (training, default: 3) |
|-m, --embed| embedding size for sequences (training, default: 200) |
|-w, --beam_width| beam width, number of predictions (testing, predicting, default: 10) |
|-i, --edit_distance| maximum edit distance of password pairs to consider during training (training, default: 3) |
|-f, --save_freq| frequency to save checkpoints of the model (training, default: 11500) |
|-k, --keep_prob| keep probability = 1 - dropout probability (training, default: 0.8) |
|-a, --password| predict passwords for this password (predicting, default: "password") |
|-j, --checkpoint| model checkpoint number to use when testing (testing, predicting, default: latest checkpoint in model dir) |
|-u, --unique_pred| number of unique predictions to generate when predicting from file (predicting, default: `beam_width`) |

## Training and Testing

Examples to start the example:

* Note: if there are checkpoints in the `/model/` dir, and the model parameters are the same, training will automatically resume from the latest checkpoint (you can choose the exact checkpoint number by editing the `checkpoint` file in the `/model/` dir with your favorite text editor).

`python pass2path_v2.py -t -q -d ./dataset_tr.csv -b 256 -i 3 -r 0.001 -k 0.6 -s 100 -e 3 -l 3`

`python pass2path_v2.py -t -d ./dataset_tr.csv -b 128 -i 2 -r 0.0003 -k 0.5 -s 1000 -e 3 -l 4 -c gru`

`python pass2path_v2.py -t -d ./dataset_tr.csv -b 50 -i 4 -r 0.001 -k 0.7 -s 1000 -e 10 -l 4 -z 100 -c gru`

`python pass2path_v2.py -t -d ./dataset_tr.csv -b 10 -i 3 -r 0.0001 -k 0.6 -s 100 -e 10 -l 3 -z 128 -m 150 -c lstm`

`python pass2path_v2.py -t -d ./dataset_tr.csv -b 10 -i 3 -r 0.0001 -k 0.6 -s 100 -e 10 -l 3 -z 128 -m 150 -c lstm -f 3000`

Model's checkpoints are saved in `/model/` dir.

## GPU
don't forget to add CUDA...
## Tensorboard
(Graphs and errors, write log)
