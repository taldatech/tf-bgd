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
* ![equation](https://latex.codecogs.com/gif.latex?%24%5Cepsilon_i%24) - a Random Variable (RV) sampled from $N(0,1)$
* $\theta$ - the weights which we wish to find their posterior distribution
* $\phi = (\mu,\sigma)$ - the parameters which serve as a condition for the distribution of $\theta$
* $\mu$ - the mean of the weights' distribution, initially sampled from $N(0,\frac{2}{n_{input} + n_{output}})$
* $\sigma$ - the STD (Variance's root) of the weights' distribution, initially set to a small constant.
* $K$ - the number of sub-networks
* $\eta$ - hyper-parameter to compenstate for the accumulated error (tunable).
* $L(\theta)$ - Loss function

Algorithm Sketch:

* Initialize: $\mu, \sigma, \eta, K$
* For each sub-network k: sample $\epsilon_0^k$ and set $\theta_0^k = \mu_0 + \epsilon_0^k \sigma_0$
* Repeat:

    1. For each sub-network k: sample $\epsilon_i^k$, compute gradients: $\frac{\partial L(\theta)}{\partial \theta_i}$
    2. Set $\mu_i \leftarrow \mu_i - \eta\sigma_i^2\mathbb{E}_{\epsilon}[\frac{\partial L(\theta)}{\partial \theta_i}] $
    3. Set $\sigma_i \leftarrow \sigma_i\sqrt{1 + (\frac{1}{2} \sigma_i\mathbb{E}_{\epsilon}[\frac{\partial L(\theta)}{\partial \theta_i}\epsilon_i])^2} - \frac{1}{2}\sigma_i^2\mathbb{E}_{\epsilon}[\frac{\partial L(\theta)}{\partial \theta_i}\epsilon_i] $
    4. Set $\theta_i^k = \mu_i + \epsilon_i^k \sigma_i$ for each k (sub-network)
    
* Until convergence criterion is met
* Note: i is the $i^{th}$ component of the vector, that is, if we have n paramaters (weights, bias) for each sub-network, then for each parameter we have $\mu_i$ and $\sigma_i$

The expectactions are estimated using Monte Carlo method:

$\mathbb{E}_{\epsilon}[\frac{\partial L(\theta)}{\partial \theta_i}] \approx \frac{1}{K}\sum_{k=1}^{K}\frac{\partial L(\theta^{(k)})}{\partial \theta_i} $


$\mathbb{E}_{\epsilon}[\frac{\partial L(\theta)}{\partial \theta_i}\epsilon_i] \approx \frac{1}{K}\sum_{k=1}^{K}\frac{\partial L(\theta^{(k)})}{\partial \theta_i}\epsilon_i^{(k)} $

### Loss Function $L(\theta)$ Derivation for Regression Problems

$L(\theta) = -log(P(D|\theta)) = -log(\prod_{i=1}^{M} P(D_i|\theta)) = -\sum_{i=1}^{M} log(P(D_i|\theta))$

Recall that from our Gaussian noise assumption, we dervied that the target (label) $t$ is also Gaussian distributed, such that: $$ P(t|x,\theta) = N(t|y(x,\theta), \beta^{-1}) $$
where $\beta$ is the percision (the inverse variance).
Assuming that the dataset is IID, we get the following:
$$ P(t|x,\theta, \beta) = \prod_{i=1}^{M} P(t_i|x_i,\theta, \beta) $$
Taking the negative logarithm, we get:
$$ -log( P(t|x,\theta, \beta)) = \frac{\beta}{2}\sum_{i=1}^M [y(x_i,\theta) - t_i]^2 -\frac{N}{2}ln(\beta) + \frac{N}{2}ln(2\pi) $$
Maximizing the log-likelihood is equivalent to minimizing the sum: $$ \frac{1}{2}\sum_{i=1}^M [y(x_i,\theta) - t_i]^2 $$ with respect to $\theta$ (looks similar to MSE, without the normalization), which is why we use `reduce_sum` in the code and not `reduce_mean`.

Note: we denote D as a general expression for the data, and in our case is the probability of the target conditiond on the input and the weights. Pay attention that $L(\theta)$ is the log of the probability which is log of an expression between [0,1], thus, the loss itself is not bounded. The probability is a Gaussian (which is of course, bounded).

### Regression using BGD

We wish to test the algorithm by learning $y = x^3$ with samples from $y = x^3 +\zeta$ such that $\zeta$~$N(0,9)$ where the loss function is MSE and eventually compare its performance to the BP algorithm which we observed in Stage One. We'll take 20 training examples and perform 40 epochs.

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
