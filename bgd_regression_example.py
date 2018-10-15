# imports
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from bgd_model import BgdModel
from matplotlib import pyplot as plt
from datetime import datetime
import time
import os
import json
import shutil
from collections import OrderedDict
from random import shuffle
import argparse

# Globals:
# write_log = False
FLAGS = tf.app.flags.FLAGS


def set_train_flags(num_sub_networks=10, hidden_units=100, num_layers=1, eta=1.0, sigma_0=0.0001,
                   batch_size=5, epochs=40, n_inputs=1, n_outputs=1, decay_steps=10000, decay_rate=1/10,
                   display_step=100, save_freq=200):
    
    tf.app.flags.FLAGS.__flags.clear()

    # Network parameters
    tf.app.flags.DEFINE_integer('num_sub_networks', num_sub_networks, 'Number of hidden units in each layer')
    tf.app.flags.DEFINE_integer('hidden_units', hidden_units, 'Number of hidden units in each layer')
    tf.app.flags.DEFINE_integer('num_layers', num_layers , 'Number of layers')

    # Training parameters
    tf.app.flags.DEFINE_float('eta', eta, 'eta parameter (step size)')
    tf.app.flags.DEFINE_float('sigma_0', sigma_0, 'Initialization for sigma parameter')
    tf.app.flags.DEFINE_integer('batch_size', batch_size, 'Batch size')
    tf.app.flags.DEFINE_integer('max_epochs', epochs, 'Maximum # of training epochs')
    tf.app.flags.DEFINE_integer('n_inputs', n_inputs, 'Inputs dimension')
    tf.app.flags.DEFINE_integer('n_outputs', n_inputs, 'Outputs dimension')
    tf.app.flags.DEFINE_integer('decay_steps', decay_steps, 'Decay steps for learning rate scheduling')
    tf.app.flags.DEFINE_float('decay_rate', decay_rate, 'Decay rate for learning rate scheduling')
    
    
    tf.app.flags.DEFINE_integer('display_freq', display_step, 'Display training status every this iteration')
    tf.app.flags.DEFINE_integer('save_freq', save_freq, 'Save model checkpoint every this iteration')


    tf.app.flags.DEFINE_string('model_dir', './model/', 'Path to save model checkpoints')
    tf.app.flags.DEFINE_string('summary_dir', './model/summary', 'Path to save model summary')
    tf.app.flags.DEFINE_string('model_name', 'linear_reg_bgd.ckpt', 'File name used for model checkpoints')
    # Ignore Cmmand Line
    tf.app.flags.DEFINE_string('w', '', '')
    tf.app.flags.DEFINE_string('s', '', '')
    tf.app.flags.DEFINE_string('e', '', '')
    tf.app.flags.DEFINE_string('b', '', '')
    tf.app.flags.DEFINE_string('n', '', '')
    tf.app.flags.DEFINE_string('l', '', '')
    tf.app.flags.DEFINE_string('t', '', '')
    tf.app.flags.DEFINE_string('g', '', '')
    tf.app.flags.DEFINE_string('f', '', '')
    tf.app.flags.DEFINE_string('r', '', '')
    tf.app.flags.DEFINE_string('k', '', '')
    tf.app.flags.DEFINE_string('y', '', '')
    tf.app.flags.DEFINE_string('u', '', '')
    tf.app.flags.DEFINE_boolean('use_fp16', False, 'Use half precision float16 instead of float32 as dtype')

    # Runtime parameters
    tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
    tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')
    
def set_predict_flags(checkpoint=-1):
    tf.app.flags.FLAGS.__flags.clear()
    latest_ckpt = tf.train.latest_checkpoint('./model/')

    if (checkpoint == -1):
        ckpt = latest_ckpt
    else:
        ckpt = './model/linear_reg_bgd.ckpt-' + str(checkpoint)
    tf.app.flags.DEFINE_string('model_path',ckpt, 'Path to a specific model checkpoint.')

    # Runtime parameters
    tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
    tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

    # Ignore Cmmand Line
    tf.app.flags.DEFINE_string('w', '', '')
    tf.app.flags.DEFINE_string('s', '', '')
    tf.app.flags.DEFINE_string('e', '', '')
    tf.app.flags.DEFINE_string('b', '', '')
    tf.app.flags.DEFINE_string('n', '', '')
    tf.app.flags.DEFINE_string('l', '', '')
    tf.app.flags.DEFINE_string('t', '', '')
    tf.app.flags.DEFINE_string('g', '', '')
    tf.app.flags.DEFINE_string('f', '', '')
    tf.app.flags.DEFINE_string('r', '', '')
    tf.app.flags.DEFINE_string('k', '', '')
    tf.app.flags.DEFINE_string('y', '', '')
    tf.app.flags.DEFINE_string('u', '', '')
    
def create_model(FLAGS):

    config = OrderedDict(sorted((dict([(key,val.value) for key,val in FLAGS.__flags.items()])).items()))
    model = BgdModel(config, 'train')
   
    return model

def restore_model(session, model, FLAGS):
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if (ckpt):
        print("Found a checkpoint state...")
        print(ckpt.model_checkpoint_path)
    if (ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
        print('Reloading model parameters..')
        model.restore(session, ckpt.model_checkpoint_path)
        
    else:
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        print('Created new model parameters..')
        session.run(tf.global_variables_initializer())
        
def batch_gen(x, y, batch_size):
    if (len(x) != len(y)):
        print("Error generating batches, source and target lists do not match")
        return
    total_samples = len(x)
    curr_batch_size = 0
    x_batch = []
    y_batch = []
    for i in range(len(x)):
        if (curr_batch_size < batch_size):
            x_batch.append(x[i])
            y_batch.append(y[i])
            curr_batch_size += 1
        else:
            yield(x_batch, y_batch)
            x_batch = [x[i]]
            y_batch = [y[i]]
            curr_batch_size = 1
    yield(x_batch, y_batch)

def batch_gen_random(x, y, batch_size):
    if (len(x) != len(y)):
        print("Error generating batches, source and target lists do not match")
        return
    total_samples = len(x)
    curr_batch_size = 0
    xy = list(zip(x,y))
    shuffle(xy)
    x_batch = []
    y_batch = []
    for i in range(len(xy)):
        if (curr_batch_size < batch_size):
            x_batch.append(xy[i][0])
            y_batch.append(xy[i][1])
            curr_batch_size += 1
        else:
            yield(x_batch, y_batch)
            x_batch = [xy[i][0]]
            y_batch = [xy[i][1]]
            curr_batch_size = 1
    yield(x_batch, y_batch)

def train(X_train, y_train, X_test, y_test, write_log=False):
    avg_error_train = []
    avg_error_valid = []
    batch_size = FLAGS.batch_size
    # Create a new model or reload existing checkpoint
    model = create_model(FLAGS)

    # Initiate TF session
    with tf.Session(graph=model.graph,config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, 
                                          log_device_placement=FLAGS.log_device_placement,
                                          gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        restore_model(sess, model, FLAGS)

        input_size = X_train.shape[0] + X_test.shape[0]
        test_size = X_test.shape[0]

        total_batches = input_size // batch_size

        print("# Samples: {}".format(input_size))
        print("Total batches: {}".format(total_batches))

        # Split data to training and validation sets
        num_validation = test_size
        total_valid_batches = num_validation // batch_size
        total_train_batches = total_batches - total_valid_batches

        print("Total validation batches: {}".format(total_valid_batches))
        print("Total training batches: {}".format(total_train_batches)) 

        
        if (write_log):
            now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            root_logdir = "tf_logs"
            logdir = "{}/run-{}/".format(root_logdir, now)
            # TensorBoard-compatible binary log string called a summary
            error_summary = tf.summary.scalar('Step-Loss', model.accuracy)
            # Write summaries to logfiles in the log directory
            file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

        step_time = 0.0
        start_time = time.time()
        global_start_time = start_time

        # Training loop
        print('Training..')
        for epoch in range(FLAGS.max_epochs):
            if (model.global_epoch_step.eval() >= FLAGS.max_epochs):
                print('Training is already complete.', \
                      'current epoch:{}, max epoch:{}'.format(model.global_epoch_step.eval(), FLAGS.max_epochs))
                break
            batches_gen = batch_gen_random(X_train, y_train, batch_size)
            batch_acc_train = []
            batch_acc_test = []
            for batch_i, batch in enumerate(batches_gen):
                X_batch = batch[0]
                Y_batch = batch[1]
                # Execute a single training step
                batch_acc_train = model.train(sess, X_batch, Y_batch)
                batch_acc_test = model.calc_accuracy(sess, X_test, y_test)
                if (write_log):
                    summary_str = error_summary.eval(feed_dict={model.inputs: X_batch, model.targets: Y_batch})
                    file_writer.add_summary(summary_str, model.global_step.eval())
                if (model.global_step.eval() % FLAGS.display_freq == 0):
                    time_elapsed = time.time() - start_time
                    step_time = time_elapsed / FLAGS.display_freq
                    print("Epoch: ", model.global_epoch_step.eval(),
                          "Batch: {}/{}".format(batch_i, total_train_batches),
                          "Train Mean Error:", batch_acc_train,
                          "Valid Mean Error:", batch_acc_test)
                # Save the model checkpoint
                if (model.global_step.eval() % FLAGS.save_freq == 0):
                    print('Saving the model..')
                    checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                    model.save(sess, checkpoint_path, global_step=model.global_step)
                    json.dump(model.config,
                              open('%s-%d.json' % (checkpoint_path, model.global_step.eval()), 'w'),
                              indent=2)
            # Increase the epoch index of the model
            model.global_epoch_step_op.eval()
            print('Epoch {0:} DONE'.format(model.global_epoch_step.eval()))
            avg_error_train.append(np.mean(batch_acc_train))
            avg_error_valid.append(np.mean(batch_acc_test))
        if (write_log):
            file_writer.close()
        print('Saving the last model..')
        checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
        model.save(sess, checkpoint_path, global_step=model.global_step)
        json.dump(model.config,
                  open('%s-%d.json' % (checkpoint_path, model.global_step.eval()), 'w'),
                  indent=2)
        total_time = time.time() - global_start_time
        print('Training Terminated, Total time: {} seconds'.format(total_time))
        return avg_error_train, avg_error_valid

def load_config(FLAGS):
    
    config = json.load(open('%s.json' % FLAGS.model_path, 'r'))
    for key, value in FLAGS.__flags.items():
        config[key] = value.value

    return config

def load_model(config):
    
    model = BgdModel(config, 'predict')
    return model

def restore_model_predict(session, model):
    if tf.train.checkpoint_exists(FLAGS.model_path):
        print('Reloading model parameters..')
        model.restore(session, FLAGS.model_path)
    else:
        raise ValueError('No such file:[{}]'.format(FLAGS.model_path))

def predict(inputs):
    # Load model config
    config = load_config(FLAGS)
    # Load configured model
    model = load_model(config)
    with tf.Session(graph=model.graph,config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, 
                                          log_device_placement=FLAGS.log_device_placement,
                                          gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        # Reload existing checkpoint
        restore_model_predict(sess, model)
        
        print("Predicting results for inputs...")
        # Prepare results dict
        results = {}
        # Predict
        results['predictions'] = model.predict(sess, inputs)
        # Statistics
        results['max_out'] = model.max_output.eval(feed_dict={model.inputs: inputs})
        results['min_out'] = model.min_output.eval(feed_dict={model.inputs: inputs})
        upper_confidence, lower_confidence = model.calc_confidence(sess, inputs)
        results['upper_confidence'] = upper_confidence
        results['lower_confidence'] = lower_confidence
        results['avg_sigma'] = np.mean([s.eval() for s in model.sigma_s])
        print("Finished predicting.")
    return results



def main():

    parser = argparse.ArgumentParser(
        description="train and test BGD regression of y=x^3")
    parser.add_argument("-w", "--write_log", help="save log for tensorboard",
                        action="store_true")
    parser.add_argument("-u", "--reset", help="reset, start training from scratch",
                        action="store_true")
    parser.add_argument("-s", "--step", type=int,
                        help="display step to show training progress, default: 10")
    parser.add_argument("-k", "--num_sub_nets", type=int,
                        help="number of sub networks (K parameter), default: 10")
    parser.add_argument("-e", "--epochs", type=int,
                        help="number of epochs to run, default: 40")
    parser.add_argument("-b", "--batch_size", type=int,
                        help="batch size, default: 1")
    parser.add_argument("-n", "--neurons", type=int,
                        help="number of hidden units, default: 100")
    parser.add_argument("-l", "--layers", type=int,
                        help="number of layers in each rnn, default: 1")
    parser.add_argument("-t", "--eta", type=float,
                        help="eta parameter ('learning rate'), deafult: 50.0")
    parser.add_argument("-g", "--sigma", type=float,
                        help="sigma_0 parameter, default: 0.002")
    parser.add_argument("-f", "--save_freq", type=int,
                        help="frequency to save checkpoints of the model, default: 200")
    parser.add_argument("-r", "--decay_rate", type=float,
                        help="decay rate of eta (exponential scheduling), default: 1/10")
    parser.add_argument("-y", "--decay_steps", type=int,
                        help="decay steps fof eta (exponential scheduling), default: 10000")
    args = parser.parse_args()

    # Prepare the dataset
    input_size = 25
    train_size = (np.ceil(0.8 * input_size)).astype(np.int)
    test_size = input_size - train_size
    # Generate dataset

    X = np.random.uniform(low=-4, high=4, size=input_size)
    y = np.power(X,3) + np.random.normal(0, 3, size=input_size)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    y_original = np.power(X, 3)
    X_sorted = X[X.argsort()]
    y_orig_sorted = y_original[X.argsort()]

    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    X_train = X_train.reshape(-1,1)
    X_test = X_test.reshape(-1,1)
    X_real_test = np.linspace(-6, 6, 2000)
    X_real_test = X_real_test.reshape(-1,1)

    if (args.write_log):
        write_log = True
    else:
        write_log = False
    if (args.step):
        display_step = args.step
    else:
        display_step = 10
    if (args.num_sub_nets):
        K = args.num_sub_nets
    else:
        K = 10
    if (args.epochs):
        epochs = args.epochs
    else:
        epochs = 40
    if (args.batch_size):
        batch_size = args.batch_size
    else:
        batch_size = 1
    if (args.neurons):
        num_units = args.neurons
    else:
        num_units = 100
    if (args.layers):
        num_layers = args.layers
    else:
        num_layers = 1
    if (args.eta):
        eta = args.eta
    else:
        eta = 50.0
    if (args.sigma):
        sigma = args.sigma
    else:
        sigma = 0.002
    if (args.save_freq):
        save_freq = args.save_freq
    else:
        save_freq = 200
    if (args.decay_rate):
        decay_rate = args.decay_rate
    else:
        decay_rate = 1/10
    if (args.decay_steps):
        decay_steps = args.decay_steps
    else:
        decay_steps = 10000
    if (args.reset):
        try:
            shutil.rmtree('./model/')
        except FileNotFoundError:
            pass

    set_train_flags(num_sub_networks=K, hidden_units=num_units, num_layers=num_layers, eta=eta, sigma_0=sigma,
        batch_size=batch_size, epochs=epochs, n_inputs=1, n_outputs=1, decay_steps=decay_steps, decay_rate=decay_rate,
            display_step=display_step, save_freq=save_freq)
    avg_error_train, avg_error_valid = train(X_train, y_train, X_test, y_test, write_log=write_log)
    set_predict_flags()
    y_real_test_res = predict(X_real_test)
    print("Maximum uncertainty: ",abs(max(y_real_test_res['upper_confidence'])))
    # Visualize Error:
    # plt.rcParams['figure.figsize'] = (15,20)
    # SSE
    plt.subplot(2,1,1)
    plt.plot(range(len(avg_error_train)), avg_error_train, label="Train")
    plt.plot(range(len(avg_error_valid)), avg_error_valid, label="Valid")
    plt.xlabel('Epoch')
    plt.ylabel('Mean Error')
    plt.title('Train and Valid Mean Error vs Epoch')
    plt.legend()
    plt.subplot(2,1,2)
    # Predictions of train and test vs original
    X_train_sorted = X_train[X_train.T.argsort()]

    y_noisy_sorted = y[X.argsort()]
    y_real = np.power(X_real_test, 3)

    plt.scatter(X_sorted, y_noisy_sorted, label='Noisy data', c='k')
    plt.plot(X_sorted, y_orig_sorted, linestyle='-', marker='o', label='True data')
    plt.plot(X_real_test, y_real_test_res['predictions'], linestyle='-', label= 'Test prediction')
    plt.plot(X_real_test, y_real, linestyle='-', label= 'y = x^3')
    low_conf = y_real_test_res['predictions'][:,0] + 100 * y_real_test_res['lower_confidence'][:,0]
    up_conf = y_real_test_res['predictions'][:,0] + 100 * y_real_test_res['upper_confidence'][:,0]
    plt.fill_between(X_real_test[:,0], low_conf, up_conf, interpolate=True, color='pink', alpha=0.5)
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(('$y=x^3$ for original input and BP predictions for noisy input'))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()