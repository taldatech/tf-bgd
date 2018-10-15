'''
Bayesian Gradient Descent
Implementation of the BGD algorithm:
The basic assumption is that in each step, the previous posterior distribution is used as the new prior distribution and that the parametric distribution is approximately a Diagonal Gaussian, 
that is, all the parameters of the weight vector `theta` are independent.

We define the following:
* `epsilon_i` - a Random Variable (RV) sampled from N(0,1)
* `theta` - the weights which we wish to find their posterior distribution
* `phi` = (mu,sigma) - the parameters which serve as a condition for the distribution of `theta`
* `mu` - the mean of the weights' distribution, initially sampled from `N(0,2/{n_input + n_output}})`
* `sigma` - the STD (Variance's root) of the weights' distribution, initially set to a small constant.
* `K` - the number of sub-networks
* `eta` - hyper-parameter to compenstate for the accumulated error (tunable).
* `L(theta)` - Loss function

* See Jupter Notebook for more details and derivations
'''
import tensorflow as tf
import numpy as np
from datetime import datetime

class BgdModel():
    def __init__(self, config, mode):

        '''
        mode: train or predict
        config: dictionary consisting of network's parameters
        config uses tf's flags
        '''

        assert mode.lower() in ['train', 'predict']

        self.config = config
        self.mode = mode.lower()

        self.num_sub_networks = config['num_sub_networks'] # K
        self.num_layers = config['num_layers']
        self.n_inputs = config['n_inputs']
        self.n_outputs = config['n_outputs']
        self.hidden_units = config['hidden_units']
        self.sigma_0 = config['sigma_0']
        self.eta = config['eta']
        self.batch_size = config['batch_size']
        # Learning Rate Scheduling:
        self.decay_steps = config['decay_steps']
        self.decay_rate = config['decay_rate']

        self.dtype = tf.float16 if config['use_fp16'] else tf.float32 # for faster learning

 
        self.build_model()

    def build_model(self):
        '''
        Builds the BNN model.
        '''
        print("building model..")

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.init_placeholders()
            self.build_variables()
            self.build_dnn()
            self.build_losses()
            self.build_grads()
            self.build_eval()
            self.build_predictions()

        # Merge all the training summaries
        self.summary_op = tf.summary.merge_all()

    def init_placeholders(self):
        '''
        Initialize the place holders to ineract with the outside world.
        '''
        print("initializing placeholders...")
        # inputs: [batch_size, data]
        self.inputs = tf.placeholder(tf.float32, shape=(None,self.n_inputs), name="inputs")

        # outputs: [batch_size, data]
        self.targets = tf.placeholder(tf.float32, shape=(None,self.n_outputs), name="outputs")


    def build_variables(self):
        '''
        Builds the variables used in the network, trainable and random-variables.
        '''
        print("building variables...")
        with tf.name_scope("variables"):
            self.global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.float32)
            self.global_step_op = \
	        tf.assign(self.global_step, self.global_step + 1)
            self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
            self.global_epoch_step_op = \
	        tf.assign(self.global_epoch_step, self.global_epoch_step + 1)
            
            # learning rate:
            self.eta_rate = tf.train.exponential_decay(np.float32(self.eta), self.global_step,
                                                     self.decay_steps, self.decay_rate)

            self.mu_s = self.build_mu_s()
            self.sigma_s = self.build_sigma_s()
            self.epsilons_s = self.build_epsilons_s()
            self.theta_s = self.build_theta_s()
            self.num_weights = (self.n_inputs + 1) * self.hidden_units + \
                                 (self.hidden_units + 1) * (self.hidden_units) * (self.num_layers - 1) + \
                                    (self.hidden_units + 1) * self.n_outputs

    def build_mu_layer(self, n_inputs, n_outputs, n_outputs_connections, name=None):
        '''
        This function creates the trainable mean variables for a layer
        '''
        if name is not None:
            name_ker = "mu_ker_" + name
            name_bias = "mu_bias_" + name
        else:
            name_ker = "mu_ker"
            name_bias = "mu_bias"
        # Reminder: we add 1 because of the bias 
        mu_ker = tf.Variable(tf.random_normal(shape=(n_inputs, n_outputs), mean=0.0, 
                                            stddev=(tf.sqrt(2 / (n_inputs + 1 + n_outputs_connections)))
                                            ),name=name_ker,trainable=False)
        mu_bias = tf.Variable(tf.random_normal(shape=(n_outputs,), mean=0.0,
                                            stddev=(tf.sqrt(2 / (n_inputs + 1 + n_outputs_connections)))
                                            ), name=name_bias, trainable=False)
        return mu_ker, mu_bias

    def build_mu_s(self):
        '''
        This function builds the mean variables for the whole network.
        Returns a list of mean variables.
        '''
        mu_s = []
        for i in range(self.num_layers + 1):
            if not i:
                # This might be wrong, since for one layer there should be one output. so
                # instead of n_hidden we should change to `n_input of next layer`
                if ( i + 1 == self.num_layers):
                    mu_ker, mu_bias = self.build_mu_layer(self.n_inputs, self.hidden_units, self.n_outputs, name="hid_0")
                else:
                    mu_ker, mu_bias = self.build_mu_layer(self.n_inputs, self.hidden_units, self.hidden_units, name="hid_0")
            elif (i == self.num_layers):
                mu_ker, mu_bias = self.build_mu_layer(self.hidden_units, self.n_outputs, self.n_outputs, name="out")
            else:
                if ( i + 1 == self.num_layers):
                    mu_ker, mu_bias = self.build_mu_layer(self.hidden_units, self.hidden_units, self.n_outputs, name="hid_" + str(i))
                else:
                    mu_ker, mu_bias = self.build_mu_layer(self.hidden_units, self.hidden_units, self.hidden_units, name="hid_" + str(i))
            mu_s += [mu_ker, mu_bias]
        return mu_s

    def build_sigma_layer(self, n_inputs, n_outputs, sigma_0=0.001 ,name=None):
        '''
        This function creates the trainable variance variables for a layer
        '''
        if name is not None:
            name_ker = "sigma_ker_" + name
            name_bias = "sigma_bias_" + name
        else:
            name_ker = "sigma_ker"
            name_bias = "sigma_bias"
        sigma_ker = tf.Variable(tf.fill((n_inputs, n_outputs), sigma_0), name=name_ker, trainable=False)
        sigma_bias = tf.Variable(tf.fill((n_outputs,), sigma_0), name=name_bias, trainable=False)
        return sigma_ker, sigma_bias

    def build_sigma_s(self):
        '''
        This function builds the variance variables for the whole network.
        Returns a list of variance variables.
        '''
        sigma_s = []
        for i in range(self.num_layers + 1):
            if not i:
                sigma_ker, sigma_bias = self.build_sigma_layer(self.n_inputs, self.hidden_units, sigma_0=self.sigma_0 ,name="hid_0")
            elif (i == self.num_layers):
                sigma_ker, sigma_bias = self.build_sigma_layer(self.hidden_units, self.n_outputs, sigma_0=self.sigma_0, name="out")
            else:
                sigma_ker, sigma_bias = self.build_sigma_layer(self.hidden_units, self.hidden_units, sigma_0=self.sigma_0, name="hid_" + str(i))
            sigma_s += [sigma_ker, sigma_bias]
        return sigma_s

    def build_epsilons_layer(self, n_inputs, n_outputs, K, name=None):
        '''
        This function creates the epsilons random variables for a layer in each sub-network k
        '''
        if name is not None:
            name_ker = "epsilons_ker_" + name
            name_bias = "epsilons_bias_" + name
        else:
            name_ker = "epsilons_ker"
            name_bias = "epsilons_bias"
        epsilons_ker = [tf.random_normal(shape=(n_inputs, n_outputs), mean=0.0, stddev=1,
                                             name=name_ker + "_" + str(i)) for i in range(K)]
        epsilons_bias = [tf.random_normal(shape=(n_outputs,), mean=0.0, stddev=1,
                                              name=name_bias + "_" + str(i)) for i in range(K)]
        return epsilons_ker, epsilons_bias

    def build_epsilons_s(self):
        '''
        This function builds the epsilons random variables for the whole network.
        Returns a list of lists of epsilons variables.
        '''
        epsilons_s = []
        for i in range(self.num_layers + 1):
            if not i:
                epsilons_ker, epsilons_bias = self.build_epsilons_layer(self.n_inputs, self.hidden_units, self.num_sub_networks ,name="hid_0")
            elif (i == self.num_layers):
                epsilons_ker, epsilons_bias = self.build_epsilons_layer(self.hidden_units, self.n_outputs, self.num_sub_networks, name="out")
            else:
                epsilons_ker, epsilons_bias = self.build_epsilons_layer(self.hidden_units, self.hidden_units, self.num_sub_networks, name="hid_" + str(i))
            epsilons_s += [epsilons_ker, epsilons_bias]
        return epsilons_s

    def build_theta_layer(self, mu, sigma, epsilons, K, name=None):
        '''
        This function creates the thea variables for a layer in each sub-network k.
        Indices for mu, sigma, epsilons:
        0 - kernel
        1 - bias
        '''
        if name is not None:
            name_ker = "theta_ker_" + name
            name_bias = "theta_bias_" + name
        else:
            name_ker = "theta_ker"
            name_bias = "theta_bias"
        
        theta_ker = [tf.identity(mu[0] + tf.multiply(epsilons[0][j], sigma[0]),
                                         name=name_ker + "_" + str(j)) for j in range(K)]
        theta_bias = [tf.identity(mu[1] + tf.multiply(epsilons[1][j], sigma[1]),
                                         name=name_bias + "_" + str(j)) for j in range(K)]
        return theta_ker, theta_bias

    def build_theta_s(self):
        '''
        This function builds the theta variables for the whole network.
        Returns a list of lists of theta variables.
        '''
        theta_s = []
        for i in range(0, 2 * (self.num_layers + 1) ,2):
            if (i == 2 * self.num_layers):
                theta_ker, theta_bias = self.build_theta_layer(self.mu_s[i:i + 2],
                                                              self.sigma_s[i:i + 2],
                                                             self.epsilons_s[i:i + 2], 
                                                             self.num_sub_networks,
                                                            name="out")
            else:
                theta_ker, theta_bias = self.build_theta_layer(self.mu_s[i:i + 2],
                                                              self.sigma_s[i:i + 2],
                                                             self.epsilons_s[i:i + 2],
                                                            self.num_sub_networks,
                                                            name="hid_" + str(i))
            theta_s += [theta_ker, theta_bias]
        return theta_s

    def build_theta_layer_boundries(self, mu, sigma, K, name=None):
        '''
        This function creates the max and min thea variables for a layer in each sub-network k.
        Indices for mu, sigma, epsilons:
        0 - kernel
        1 - bias
        '''
        if name is not None:
            name_ker = "theta_ker_" + name
            name_bias = "theta_bias_" + name
        else:
            name_ker = "theta_ker"
            name_bias = "theta_bias"
        
        theta_ker_max = [tf.identity(mu[0] + sigma[0],
                                         name=name_ker + "_max_" + str(j)) for j in range(K)]
        theta_bias_max = [tf.identity(mu[1] + sigma[1],
                                         name=name_bias + "_max_" + str(j)) for j in range(K)]
    
        theta_ker_min = [tf.identity(mu[0] - sigma[0],
                                         name=name_ker + "_min_" + str(j)) for j in range(K)]
        theta_bias_min = [tf.identity(mu[1] - sigma[1],
                                         name=name_bias + "_min_" + str(j)) for j in range(K)]
    
        return theta_ker_min, theta_bias_min, theta_ker_max, theta_bias_max

    def build_theta_s_boundries(self):
        '''
        This function builds the max and min theta variables for the whole network.
        Returns a list of lists of theta variables.
        '''
        theta_s_min = []
        theta_s_max = []
        for i in range(0, 2 * (self.num_layers + 1) ,2):
            if (i == 2 * self.num_layers):
                theta_ker_min, theta_bias_min, theta_ker_max, theta_bias_max = self.build_theta_layer_boundries(self.mu_s[i:i + 2],
                                                                                                                self.sigma_s[i:i + 2],
                                                                                                                self.num_sub_networks,
                                                                                                               name="out")
            else:
                theta_ker_min, theta_bias_min, theta_ker_max, theta_bias_max = self.build_theta_layer_boundries(self.mu_s[i:i + 2],
                                                                                                                self.sigma_s[i:i + 2] ,
                                                                                                                self.num_sub_networks,
                                                                                                                name="hid_" + str(i))
            theta_s_min += [theta_ker_min, theta_bias_min]
            theta_s_max += [theta_ker_max, theta_bias_max]
        return theta_s_min, theta_s_max

    def build_hidden_layers(self, inputs, n_layers, n_hidden, K, activation=tf.nn.relu):
        '''
        This function builds and denses the hidden layers of the network.
        Returns the layers and their corresponding outputs.
        '''
        hiddens_func = []
        hiddens_out = []
        for i in range(n_layers):
            if not i:
                hid_funcs = [tf.layers.Dense(n_hidden, name="hidden_0_" + str(k), activation=activation) for k in range(K)]
                hid_out = [hid_funcs[k](inputs) for k in range(K)]
                hiddens_func.append(hid_funcs)
                hiddens_out.append(hid_out)
            else:
                hid_funcs = [tf.layers.Dense(n_hidden, name="hidden_" + str(i) + "_" + str(k),
                                            activation=activation) for k in range(K)]
                hid_out = [hid_funcs[k](hiddens_out[i - 1][k]) for k in range(K)]
                hiddens_func.append(hid_funcs)
                hiddens_out.append(hid_out)
        return hiddens_func, hiddens_out

    def build_dnn(self):
        '''
        This function builds the deep network's layout in terms of layers.
        '''
        print("building layers...")     
        with tf.name_scope("dnns"):
            self.hiddens_funcs, self.hiddens_out = self.build_hidden_layers(self.inputs,
                                                                            self.num_layers,
                                                                            self.hidden_units,
                                                                            self.num_sub_networks)
            self.out_funcs = [tf.layers.Dense(self.n_outputs, name="outputs_" + str(i), activation=None) \
                                                     for i in range(self.num_sub_networks)]
            self.outputs = [self.out_funcs[k](self.hiddens_out[-1][k]) for k in range(self.num_sub_networks)]
        total_hidden_params = sum([self.hiddens_funcs[i][0].count_params() for i in range(self.num_layers)])
        graph_params_count = total_hidden_params + self.out_funcs[0].count_params()
        if (graph_params_count != self.num_weights):
         print("Number of actual parameters ({}) different from the calculated number ({})".format(
             graph_params_count, self.num_weights))

    def build_losses(self):
        '''
        This functions builds the error and losses of the network.
        '''
        print("configuring loss...")
        with tf.name_scope("loss"):
            errors = [(self.outputs[i] - self.targets) for i in range(self.num_sub_networks)]
            self.losses = [0.5 * tf.reduce_sum(tf.square(errors[i]), name="loss_" + str(i)) \
                                                             for i in range(self.num_sub_networks)]

    def grad_mu_sigma(self, gradients_tensor, mu, sigma, epsilons, eta):
        # Calculate number of sub-networks = samples:
        K = len(epsilons[0])
        '''
        We need to sum over K, that is, for each weight in num_weights, we calculate
        the average/weighted average over K.
        gradients_tensor[k] is the gradients of sub-network k out of K.
        Note: in order to apply the gradients later, we should keep the variables in gradient_tensor apart.
        '''
        # Number of separated variables in each network (in order to update each one without changing the shape)
        num_vars = sum(1 for gv in gradients_tensor[0] if gv[0] is not None)
        mu_n = []
        sigma_n = []
        # filter non-relavent variables
        for k in range(len(gradients_tensor)):
            gradients_tensor[k] = [gradients_tensor[k][i] for i in range(len(gradients_tensor[k]))
                                if gradients_tensor[k][i][0] is not None]
        for var_layer in range(num_vars):
            var_list = [tf.reshape(gradients_tensor[k][var_layer][0], [-1]) for k in range(K)]
            E_L_theta = tf.reduce_mean(var_list, axis=0)
            var_list = [tf.reshape(gradients_tensor[k][(var_layer)][0] * epsilons[var_layer][k], [-1]) for k in range(K)]
            E_L_theta_epsilon = tf.reduce_mean(var_list, axis=0)
            # reshape it back to its original shape
            new_mu = mu[var_layer] - eta * tf.square(sigma[var_layer]) * tf.reshape(E_L_theta, mu[var_layer].shape)
            mu_n.append(new_mu)
            E_L_theta_epsilon = tf.reshape(E_L_theta_epsilon, sigma[var_layer].shape)
            new_sigma = sigma[var_layer] * tf.sqrt(1 + tf.square(0.5 * sigma[var_layer] * E_L_theta_epsilon)) - 0.5 * tf.square(sigma[var_layer]) * E_L_theta_epsilon
            sigma_n.append(new_sigma)
        return mu_n, sigma_n

    def build_grads(self):
        '''
        This functions builds the gradients update nodes of the network.
        '''
        print("configuring optimization and gradients...")
        with tf.name_scope("grads"):
            optimizer = tf.train.GradientDescentOptimizer(self.eta)
            gradients = [optimizer.compute_gradients(loss=self.losses[i]) for i in range(self.num_sub_networks)]
            mu_n, sigma_n = self.grad_mu_sigma(gradients, self.mu_s, self.sigma_s, self.epsilons_s, self.eta_rate)
            self.grad_op = [self.mu_s[i].assign(mu_n[i]) for i in range(len(self.mu_s))] + \
                             [self.sigma_s[i].assign(sigma_n[i]) for i in range(len(self.sigma_s))]

    def build_eval(self):
        '''
        This function builds the model's evaluation nodes.
        '''
        print("preparing evaluation...")
        with tf.name_scope("eval"):
            self.accuracy = tf.reduce_mean([tf.reduce_mean(self.losses[i]) for i in range(self.num_sub_networks)])
    
    def build_predictions(self):
        '''
        This function builds the model's prediction nodes.
        '''
        print("preparing predictions")
        with tf.name_scope("prediction"):
            self.predictions = tf.reduce_mean(self.outputs, axis=0)
            self.mean, self.variance = tf.nn.moments(tf.convert_to_tensor(self.outputs), axes=[0])
            self.std = tf.sqrt(self.variance)
            self.max_output = tf.reduce_max(self.outputs, axis=0)
            self.min_output = tf.reduce_min(self.outputs, axis=0)

    def weights_init(self, sess):
        '''
        Initialize BNN weights.
        '''
        for k in range(self.num_sub_networks):
            weights_init = [self.theta_s[i][k].eval() for i in range(len(self.theta_s))]
            for i in range(self.num_layers):
                self.hiddens_funcs[i][k].set_weights([weights_init[2 * i], weights_init[2 * i + 1]])
            self.out_funcs[k].set_weights([weights_init[-2], weights_init[-1]])
        
    def train(self, sess, inputs, outputs):
        '''
        Execute a single training step.
        Returns train step accuracy.
        '''
        sess.run(self.grad_op, feed_dict={self.inputs: inputs, self.targets: outputs})
        sess.run(self.global_step_op)
        for k in range(self.num_sub_networks):
            weights_calc = [self.theta_s[i][k].eval() for i in range(len(self.theta_s))]
            for i in range(self.num_layers):
                self.hiddens_funcs[i][k].set_weights([weights_calc[2 * i], weights_calc[2 * i + 1]])
            self.out_funcs[k].set_weights([weights_calc[-2], weights_calc[-1]])
        acc_train = self.accuracy.eval(feed_dict={self.inputs: inputs, self.targets: outputs})
        return acc_train

    def calc_accuracy(self, sess, inputs, outputs):
        '''
        Returns the accuracy over the inputs using the BNN's current weights.
        '''
        return self.accuracy.eval(feed_dict={self.inputs: inputs, self.targets: outputs})

    def predict(self, sess, inputs):
        '''
        Returns predictions for the inputs using the BNN's current weights.
        '''
        return self.predictions.eval(feed_dict={self.inputs: inputs})

    def calc_confidence(self, sess, inputs):
        '''
        Returns the upper and lower confidence for the inputs using the BNN's current weights. 
        '''
        stan_dv = self.std.eval(feed_dict={self.inputs: inputs})
        upper_conf = stan_dv
        lower_conf = -1 * stan_dv
        return upper_conf, lower_conf

    def save(self, sess, path, var_list=None, global_step=None):
        # var_list = None returns the list of all saveable variables
        saver = tf.train.Saver(var_list)

        save_path = saver.save(sess, save_path=path, global_step=global_step)
        print('model saved at %s' % save_path)
        

    def restore(self, sess, path, var_list=None):
        # var_list = None returns the list of all saveable variables
        saver = tf.train.Saver(var_list)
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)
