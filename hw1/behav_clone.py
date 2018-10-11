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

"""
Code to load an expert policy and generate a behavioral cloning model.

Example usage:
    python behav_clone.py expert_data/Ant-v2.pkl Ant-2.pkl --param xxx

Author of this script: Nathan Lambert nol@berkeley.edu
"""


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('qpart', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
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
    # Runs part two of the question, training behavioral cloning
    if args.qpart == 'two':
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
        depth = 3
        hidden_width = 200
        act_fnc = 'ReLU'
        epochs = 200
        lossfnc = 'MSE'
        learning_rate = 1e-6
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

        # Initializing the variables
        init = tf.global_variables_initializer()

        # printing variable
        display_step = 10

        saver = tf.train.Saver()

        with tf.Session(config=tf.ConfigProto(use_per_session_threads=True)) as sess:

            # intitialize variables
            sess.run(init)

            TRAIN = True
            # Trains or....
            if TRAIN:
                # Training cycle
                for ep in range(epochs):
                    avg_loss = 0.
                    total_batches = int(train_cnt/batch)
                    test_batches = int(test_cnt/batch)
                    # Loop over all batches
                    for i in range(total_batches):
                        batch_idx = np.random.randint(0,train_cnt,size=batch)

                        # data_batch, label_batch = next_batch(batch, obs_train, act_train)

                        loss, _ = sess.run([loss_op, train_op],
                            feed_dict={obs_place: obs_train[batch_idx, ], act_place: act_train[batch_idx, ]})

                        avg_loss += loss / total_batches

                    test_loss = 0
                    for i in range(test_batches):
                        data_batch = obs_test[i*batch:(i+1)*batch,:]
                        label_batch = act_test[i*batch:(i+1)*batch,:]
                        loss = sess.run([loss_op],
                            feed_dict={obs_place: data_batch, act_place: label_batch})
                        test_loss += loss[0] / test_batches

                    # Display logs per epoch step
                    if ep % display_step == 0:
                        print("Epoch:", '%04d' % (ep), "training loss={:.3f}, ".format(avg_loss), "Test loss={:.3f}".format(test_loss))
                        saver.save(sess, 'chkpts/'+ args.envname + '-trained_variables.ckpt', global_step=ep)

                print("Optimization Finished!")
                date_str = str(datetime.datetime.now())[:-7]
                date_str = date_str.replace(' ','--').replace(':', '-')
                export_dir = 'models/' + args.envname + '-' + date_str + 'train.pbtxt'
                print("Model saved in path: %s" % export_dir)
                tf.train.write_graph(sess.graph_def, 'models/', args.envname + '-' + date_str + 'train.pbtxt')

            # loads
            else:
                print('Loading Existing Model')
                path = 'DIRECTORY'


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

            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))

            expert_data = {'observations': np.array(observations),
                           'actions': np.array(actions)}

            # with open(os.path.join('expert_data', args.envname + '.pkl'), 'wb') as f:
            #     pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)


    # Runs part three of the question, hyperparamter tuning
    elif args.qpart == 'three':
        print('Running part 3 of the problem...')


        # sweep network width
        repeat = 4
        params = [10,50, 100, 200, 400]
        params = sorted(params*repeat)
        with open(str('logs/'+args.envname+'.txt'), 'w') as f:
            for p in params:
                with open(os.path.join('expert_data/', args.envname + '.pkl'), 'rb') as exp:
                    unpickler = pickle.Unpickler(exp)
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
                depth = 3
                hidden_width = p
                act_fnc = 'ReLU'
                epochs = 200
                lossfnc = 'MSE'
                learning_rate = 1e-6
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

                # Initializing the variables
                init = tf.global_variables_initializer()

                # printing variable
                display_step = 10

                saver = tf.train.Saver()

                with tf.Session(config=tf.ConfigProto(use_per_session_threads=True)) as sess:

                    # intitialize variables
                    sess.run(init)

                    TRAIN = True
                    # Trains or....
                    if TRAIN:
                        # Training cycle
                        for ep in range(epochs):
                            avg_loss = 0.
                            total_batches = int(train_cnt/batch)
                            test_batches = int(test_cnt/batch)
                            # Loop over all batches
                            for i in range(total_batches):
                                batch_idx = np.random.randint(0,train_cnt,size=batch)

                                # data_batch, label_batch = next_batch(batch, obs_train, act_train)

                                loss, _ = sess.run([loss_op, train_op],
                                    feed_dict={obs_place: obs_train[batch_idx, ], act_place: act_train[batch_idx, ]})

                                avg_loss += loss / total_batches

                            test_loss = 0
                            for i in range(test_batches):
                                data_batch = obs_test[i*batch:(i+1)*batch,:]
                                label_batch = act_test[i*batch:(i+1)*batch,:]
                                loss = sess.run([loss_op],
                                    feed_dict={obs_place: data_batch, act_place: label_batch})
                                test_loss += loss[0] / test_batches

                            # Display logs per epoch step
                            if ep % display_step == 0:
                                # print("Epoch:", '%04d' % (ep), "training loss={:.3f}, ".format(avg_loss), "Test loss={:.3f}".format(test_loss))
                                saver.save(sess, 'chkpts/'+ args.envname + '-trained_variables.ckpt', global_step=ep)

                        print("Optimization Finished!")
                        date_str = str(datetime.datetime.now())[:-7]
                        date_str = date_str.replace(' ','--').replace(':', '-')
                        export_dir = 'models/' + args.envname + '-' + date_str + 'train.pbtxt'
                        print("Model saved in path: %s" % export_dir)
                        # tf.train.write_graph(sess.graph_def, 'models/', args.envname + '-' + date_str + 'train.pbtxt')

                    # loads
                    else:
                        print('Loading Existing Model')
                        path = 'DIRECTORY'


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

                    # print('returns', returns)
                    print('mean return', np.mean(returns))
                    print('std of return', np.std(returns))

                    expert_data = {'observations': np.array(observations),
                                   'actions': np.array(actions)}

                # stores values
                print('Parameter: ' +  str(p) + '  mean:{:.3f}'.format(np.mean(returns)) + ' std:{:.3f}'.format(np.std(returns)))
                f.write('Parameter:' +  str(p) + '  mean:{:.3f}'.format(np.mean(returns)) + ' std:{:.3f}'.format(np.std(returns)) + '\n')
            print('Done with parameter sweep!')


    elif args.qpart == 'three_plot':
        vals = []
        with open('logs/Reacher-v2.txt') as csvfile:
            paramsweep = csv.reader(csvfile, delimiter=':')
            for row in paramsweep:
                p = float(row[1].partition(' ')[0])
                mean = float(row[2].partition(' ')[0])
                std = float(row[3].partition(' ')[0])
                vals.append([p,mean,std])
        vals = np.array(vals)
        ps = np.unique(vals[:,0])
        vals_avg = np.array([np.mean(vals[np.where(vals[:,0]==p)],axis=0) for p in ps])


        # plot with seaborn

        sns.set(style="darkgrid")

        df = pd.DataFrame(data=vals_avg[:,:],    # values
                        columns=['Width', 'Means', 'Std'])  # 1st row as the column names
        print(df)
        # sns.load_dataset(df)
        # Plot the responses for different events and regions
        ax = sns.lineplot( x="Width", y="Means", ci="Std",data=df)
        # ax.errorbar(df.index, mean, yerr=df["Std"], fmt='-o') #fmt=None to plot bars only
        plt.title("Rollout Mean vs Hidden Layer Width, env: " + args.envname)
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
