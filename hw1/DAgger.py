import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import datetime
import csv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

"""
Code to load an expert policy and generate a behavioral cloning model with the DAgger algorithm.

Example usage:
    python DAgger.py Ant-2.pkl experts/Ant-v2.pkl  --plot

Author of this script: Nathan Lambert nol@berkeley.edu
"""


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument( '--plot', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()

    def BehavioralClone(features, params, layers):
        # network that will fit to oberservations or actions
        # considering making the mode flag such that if mode == 'DAgger', then it will rollout its own trials

        hid = tf.layers.dense(features, units=layers[0][0], activation=layers[0][1])

        # hidden layers
        for (h, act) in layers[1:]:
            hid = tf.layers.dense(inputs=hid, units =h, activation = act)

        # output layer
        out_layer = tf.layers.dense(hid, units=dim_act) #No activation function on output layer

        return out_layer

    # log =
    print('Loading File: ', str(args.envname))

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')


    # Runs part two of the question, training behavioral cloning
    print('Running part 2 of the problem...')
    with open(os.path.join('expert_data/', args.envname + '.pkl'), 'rb') as f:
        unpickler = pickle.Unpickler(f)
        expert_data = unpickler.load()
        observations = np.squeeze(expert_data['observations'])
        actions = np.squeeze(expert_data['actions'])

    # Shape of data
    print('Shape of the datasets to use:')
    print('   Observations: ', np.shape(observations))
    print('   Actions: ', np.shape(actions))

    # Statistics of data
    # print('Observations...\n   Mean: ', np.mean(observations, axis=0), '\n   Variance: ', np.var(observations,axis=0))
    # print('Actions...\n   Mean: ', np.mean(actions, axis=0), '\n   Variance: ', np.var(actions,axis=0))

    # Setup and train network
    print('Setting up NN')
    # Some parameters of the network
    depth = 4
    hidden_width = 600
    act_fnc = 'ReLU'
    epochs = 100
    lossfnc = 'MSE'
    learning_rate = 1e-5
    batch = 50

    # package in object
    params ={ "depth" : depth,
            "hidden_width" : hidden_width,
            "activation" : act_fnc,
            "lossfnc" : lossfnc,
            "epochs" : epochs,
            "lr" : learning_rate,
            "batch" : batch
    }

    print_nn_params(params)

    # remove empty dimension
    n, dim_obs = np.shape(observations)
    _, dim_act = np.shape(actions)

    train_size = .8
    train_cnt = int(n*train_size)
    test_cnt = n-train_cnt
    obs_train = observations[:train_cnt,:]
    act_train = actions[:train_cnt,:]
    obs_test = observations[train_cnt:,:]
    act_test = actions[train_cnt:,:]


    # layers object for iteratively creating hidden layers
    layers = [
        (hidden_width, tf.nn.relu),
        (hidden_width, tf.nn.relu),
        (hidden_width, tf.nn.relu),
         (hidden_width, None)
    ]


    obs_place = tf.placeholder(tf.float32, shape=[None, dim_obs])
    act_place = tf.placeholder(tf.float32, shape=[None, dim_act])

    # Create network objects
    predictions = BehavioralClone(obs_place, params, layers)

    # Calculate loss
    loss_op = tf.losses.mean_squared_error(act_place, predictions)
    optimizer = tf.train.AdamOptimizer(learning_rate=params["lr"])
    train_op = optimizer.minimize(loss = loss_op) #, global_step = tf.train.get_global_step())

    # Initializin   g the variables
    init = tf.global_variables_initializer()

    # printing variable
    display_step = 10

    saver = tf.train.Saver()
    dagger_it = 11
    if not args.load:
        with tf.Session(config=tf.ConfigProto(use_per_session_threads=True)) as sess:
            # intitialize variables
            sess.run(init)
            # DAgger arrays
            arr_mean = []
            arr_std = []
            arr_train_size = []
            for d in range(dagger_it):
                print('Dagger Iteration: ', d)
                sess.run(init)

                print(len(obs_train), len(act_train))

                # Training cycle
                for ep in range(epochs):
                    avg_loss = 0.
                    total_batches = int(train_cnt*(d+1)/batch)
                    test_batches = int(test_cnt/batch)
                    # Loop over all batches
                    for i in range(total_batches):
                        batch_idx = np.random.randint(0,train_cnt*(d+1),size=batch)

                        # data_batch, label_batch = next_batch(batch, obs_train, act_train)

                        loss, _ = sess.run([loss_op, train_op],
                            feed_dict={obs_place: obs_train[batch_idx, ], act_place: act_train[batch_idx, ]})

                        avg_loss += loss / total_batches

                    test_loss = 0.
                    for i in range(test_batches):
                        data_batch = obs_test[i*batch:(i+1)*batch,:]
                        label_batch = act_test[i*batch:(i+1)*batch,:]
                        loss = sess.run([loss_op],
                            feed_dict={obs_place: data_batch, act_place: label_batch})
                        test_loss += loss[0] / test_batches

                    # Display logs per epoch step
                    if ep % display_step == 0:
                        print("Epoch:", '%04d' % (ep), "training loss={:.5f}, ".format(avg_loss), "Test loss={:.5f}".format(test_loss))
                        saver.save(sess, 'chkpts/'+ args.envname + '-trained_variables.ckpt', global_step=ep)

                print("...sub ptimization Finished!")
                if False:
                    date_str = str(datetime.datetime.now())[:-7]
                    date_str = date_str.replace(' ','--').replace(':', '-')
                    export_dir = 'models/' + args.envname + '-' + date_str + 'train.pbtxt'
                    print("Model saved in path: %s" % export_dir)
                    tf.train.write_graph(sess.graph_def, 'models/', args.envname + '-' + date_str + 'train.pbtxt')

                env = gym.make(args.envname)
                max_steps = args.max_timesteps or env.spec.timestep_limit

                returns = []
                observations = []
                actions = []
                for i in range(args.num_rollouts):
                    print('iter', i)
                    obs = env.reset()
                    done = False
                    totalr = 0.
                    steps = 0
                    while not done:
                        action = predictions.eval(feed_dict={obs_place: obs[None, ]})
                        observations.append(obs)
                        actions.append(action)
                        obs, r, done, _ = env.step(action)
                        totalr += r
                        steps += 1
                        if args.render:
                            env.render()
                        if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                        if steps >= max_steps:
                            break
                    returns.append(totalr)

                 # expert labeling
                act_exp = []
                for idx in range(len(observations)):
                    act_exp.append(policy_fn(observations[idx][None, :]))
                    # act_exp.append(predictions.eval(feed_dict={obs_place: observations[idx][None, :]}))

                # record training size
                dag_size = obs_train.shape[0]

                # data aggregation
                obs_train = np.concatenate((obs_train, np.array(observations)), axis=0)
                act_train = np.concatenate((act_train, np.squeeze(np.array(act_exp))), axis=0)

                # record mean return & std
                arr_mean = np.append(arr_mean, returns)
                # arr_std = np.append(arr_std, returns)
                arr_train_size = np.append(arr_train_size, [dag_size]*len(returns))

                # print('returns', returns)
                print('mean return', np.mean(returns))
                print('std of return', np.std(returns))

                # now need to merge dataset!

                new_data = {'observations': np.array(observations), 'actions': np.array(actions)}

        print(arr_mean)
        # print(arr_std)
        print(arr_train_size)
        arr_train_size = arr_train_size.reshape(-1,1)
        arr_mean = arr_mean.reshape(-1,1)
        # np.save('tmp/arrays', (arr_train_size, arr_mean))
        # if plotting, plot!hhh
    else:
        data = np.load('tmp/arrays.npz.npy')
        arr_train_size = data[0].reshape(-1,1)
        arr_mean = data[1].reshape(-1,1)
        print(np.shape(arr_mean))
        print(np.shape(arr_train_size))

    if args.plot:
        # plot with seaborn

        font = {'family' : 'normal',
        'size'   : 18}

        matplotlib.rc('font', **font)
        matplotlib.rc('lines', linewidth=2)

        sns.set(style="darkgrid")
        data = np.concatenate((arr_train_size, arr_mean), axis=1)
        df = pd.DataFrame(data=data,    # values
                        columns=['Samples', 'Returns'])  # 1st row as the column names
        print(df)
        # sns.load_dataset(df)
        # Plot the responses for different events and regions
        ax = sns.lineplot( x="Samples", y="Returns", ci="sd",data=df)
        ax.axhline(y=-9.760, label='Behavioral Clone Policy', color='r', linestyle=':')
        ax.axhline(y=-3.984, label='Expert Policy', color='g', linestyle=':')

        # ax.errorbar(df.index, mean, yerr=df["Std"], fmt='-o') #fmt=None to plot bars only
        plt.legend()
        plt.title("Rollout Mean vs Samples (Dagger Iteration), env: " + args.envname)
        plt.show()



def next_batch(num, data, labels):
    '''
    Returns a total of num random samples and labels
    '''

    idx = np.arange(0, data.get_shape().as_list()[0])
    data = data.eval().astype(np.float32)
    labels = labels.eval()
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i,:] for i in idx]
    labels_shuffle = [labels[i,:] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def print_nn_params(params):
    # Prints a list of parameters in one line of Code
    print('Network Parameters')
    for p in params:
        print('  |', p, ' is: ', params[p])


if __name__ == '__main__':
    print('\n')
    main()
    print('\n')
