"""
Original code from John Schulman for CS294 Deep Reinforcement Learning Spring 2017
Adapted for CS294-112 Fall 2017 by Abhishek Gupta and Joshua Achiam
Adapted for CS294-112 Fall 2018 by Michael Chang and Soroush Nasiriany
"""
import numpy as np
import tensorflow as tf
import gym
import logz
import os
import time
import inspect
from multiprocessing import Process

#============================================================================================#
# Utilities
#============================================================================================#

#========================================================================================#
#                           ----------PROBLEM 2----------
#========================================================================================#
def build_mlp(input_placeholder, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None):
    """
        Builds a feedforward neural network

        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            output_size: size of the output layer
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of the hidden layer
            activation: activation of the hidden layers
            output_activation: activation of the ouput layers

        returns:
            output placeholder of the network (the result of a forward pass)

        Hint: use tf.layers.dense
    """
    # Define scope
    with tf.variable_scope(scope):
        # YOUR CODE HERE
        hid = tf.layers.dense(input_placeholder, units=size, kernel_initializer=tf.orthogonal_initializer, activation=activation)

        # hidden layers
        for i in range(n_layers):
            hid = tf.layers.dense(inputs=hid, units =size, kernel_initializer=tf.orthogonal_initializer, activation = activation)

        # output layer
        output_placeholder = tf.layers.dense(hid, units=output_size, kernel_initializer=tf.orthogonal_initializer, activation = output_activation)

    return output_placeholder

def pathlength(path):
    return len(path["reward"])

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

#============================================================================================#
# Policy Gradient
#============================================================================================#

class Agent(object):
    def __init__(self, computation_graph_args, sample_trajectory_args, estimate_return_args):
        super(Agent, self).__init__()
        self.ob_dim = computation_graph_args['ob_dim']
        self.ac_dim = computation_graph_args['ac_dim']
        self.discrete = computation_graph_args['discrete']
        self.size = computation_graph_args['size']
        self.n_layers = computation_graph_args['n_layers']
        self.learning_rate = computation_graph_args['learning_rate']

        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']

        self.gamma = estimate_return_args['gamma']
        self.reward_to_go = estimate_return_args['reward_to_go']
        self.nn_baseline = estimate_return_args['nn_baseline']
        self.normalize_advantages = estimate_return_args['normalize_advantages']

    def init_tf_sess(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__() # equivalent to `with self.sess:`
        tf.global_variables_initializer().run() #pylint: disable=E1101

    #========================================================================================#
    #                           ----------PROBLEM 2----------
    #========================================================================================#
    def define_placeholders(self):
        """
            Placeholders for batch batch observations / actions / advantages in policy gradient
            loss function.
            See Agent.build_computation_graph for notation

            returns:
                sy_ob_no: placeholder for observations
                sy_ac_na: placeholder for actions
                sy_adv_n: placeholder for advantages
        """

        sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        if self.discrete:
            sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
        else:
            sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)

        # YOUR CODE HERE
        # sy_adv_n = None
        # sy_adv_n = tf.placeholder(shape=[None, self.ob_dim, self.ac_dim], name = "adv", dtype=tf.float32)
        sy_adv_n = tf.placeholder(shape=[None], name = "adv", dtype=tf.float32)
        return sy_ob_no, sy_ac_na, sy_adv_n


    #========================================================================================#
    #                           ----------PROBLEM 2----------
    #========================================================================================#
    def policy_forward_pass(self, sy_ob_no):
        """ Constructs the symbolic operation for the policy network outputs,
            which are the parameters of the policy distribution p(a|s)

            arguments:
                sy_ob_no: (batch_size, self.ob_dim)

            returns:
                the parameters of the policy.

                if discrete, the parameters are the logits of a categorical distribution
                    over the actions
                    sy_logits_na: (batch_size, self.ac_dim)

                if continuous, the parameters are a tuple (mean, log_std) of a Gaussian
                    distribution over actions. log_std should just be a trainable
                    variable, not a network output.
                    sy_mean: (batch_size, self.ac_dim)
                    sy_logstd: (self.ac_dim,)

            Hint: use the 'build_mlp' function to output the logits (in the discrete case)
                and the mean (in the continuous case).
                Pass in self.n_layers for the 'n_layers' argument, and
                pass in self.size for the 'size' argument.
        """

        if self.discrete:
            # YOUR_CODE_HERE
            # sy_logits_na = tf.placeholder(shape=[None, self.ac_dim], name = "adv", dtype=tf.float32)
            sy_logits_na = build_mlp(sy_ob_no, self.ac_dim, "Sym", self.n_layers, self.size)
            return sy_logits_na
        else:
            # YOUR_CODE_HERE
            sy_mean =  build_mlp(sy_ob_no, self.ac_dim, "Sym", self.n_layers, self.size)

            # mean, var = tf.nn.moments(x, axes=[1])
            sy_logstd = tf.Variable(tf.zeros([self.ac_dim]))
            return (sy_mean, sy_logstd)

    #========================================================================================#
    #                           ----------PROBLEM 2----------
    #========================================================================================#
    def sample_action(self, policy_parameters):
        """ Constructs a symbolic operation for stochastically sampling from the policy
            distribution

            arguments:
                policy_parameters
                    if discrete: logits of a categorical distribution over actions
                        sy_logits_na: (batch_size, self.ac_dim)
                    if continuous: (mean, log_std) of a Gaussian distribution over actions
                        sy_mean: (batch_size, self.ac_dim)
                        sy_logstd: (self.ac_dim,)

            returns:
                sy_sampled_ac:
                    if discrete: (batch_size,)
                    if continuous: (batch_size, self.ac_dim)

            Hint: for the continuous case, use the reparameterization trick:
                 The output from a Gaussian distribution with mean 'mu' and std 'sigma' is

                      mu + sigma * z,         z ~ N(0, I)

                 This reduces the problem to just sampling z. (Hint: use tf.random_normal!)
        """

        if self.discrete:
            sy_logits_na = policy_parameters
            # YOUR_CODE_HERE
            # dist = Categorical(probs=p)
            # dist.prob(0)
            dist = tf.distributions.Categorical(logits=sy_logits_na)
            if self.ac_dim == 1:
                sy_sampled_ac = dist.sample(sample_shape=[tf.shape(sy_logits_na)[0]])
                sy_sampled_ac = sy_sampled_ac
            else:
                sy_sampled_ac = dist.sample(sample_shape=[tf.shape(sy_logits_na)[0]]) #, self.ac_dim])
        else:
            sy_mean, sy_logstd = policy_parameters
            # YOUR_CODE_HERE
            z = tf.random_normal([tf.shape(sy_mean)[0], self.ac_dim],0,1)
            sy_sampled_ac = sy_mean + tf.exp(sy_logstd)*tf.random_normal(tf.shape(sy_mean)) #z #tf.multiply(z,tf.exp(sy_logstd))

        # print("Sampled Action:", sy_sampled_ac)
        return sy_sampled_ac

    #========================================================================================#
    #                           ----------PROBLEM 2----------
    #========================================================================================#
    def get_log_prob(self, policy_parameters, sy_ac_na):
        """ Constructs a symbolic operation for computing the log probability of a set of actions
            that were actually taken according to the policy

            arguments:
                policy_parameters
                    if discrete: logits of a categorical distribution over actions
                        sy_logits_na: (batch_size, self.ac_dim)
                    if continuous: (mean, log_std) of a Gaussian distribution over actions
                        sy_mean: (batch_size, self.ac_dim)
                        sy_logstd: (self.ac_dim,)

                sy_ac_na:
                    if discrete: (batch_size,)
                    if continuous: (batch_size, self.ac_dim)

            returns:
                sy_logprob_n: (batch_size)

            Hint:
                For the discrete case, use the log probability under a categorical distribution.
                For the continuous case, use the log probability under a multivariate gaussian.
        """

        if self.discrete:
            sy_logits_na = policy_parameters
            # YOUR_CODE_HERE
            dist = tf.distributions.Categorical(logits=sy_logits_na)
            # sy_logprob_n = tf.nn.softmax_cross_entropy_with_logits(labels=sy_ac_na, logits=sy_logits_na)
            sy_logprob_n = dist.log_prob(sy_ac_na)
        else:
            sy_mean, sy_logstd = policy_parameters
            # YOUR_CODE_HERE
            # NEGATIVE LIKELIHOOD OF M_GAUSSIAN:
            # -l(mu,Sig) = (N/2)*log(det(Sig)) + (1/2)*sum(i=1^N) (x_i - mu)^T Sig ^-1 (x_i - mu)
            # mt = tf.square(sy_ac_na-sy_mean)
            # neg_likelihoods =  tf.add(tf.divide(mt, tf.exp(sy_logstd)), tf.log(tf.exp(sy_logstd)))
            # # Should use dist.log_prob(value)
            multGauss = tf.contrib.distributions.MultivariateNormalDiag(loc=sy_mean,scale_diag = tf.exp(sy_logstd))
            sy_logprob_n = multGauss.log_prob(sy_ac_na)
        return sy_logprob_n

    def build_computation_graph(self):
        """
            Notes on notation:

            Symbolic variables have the prefix sy_, to distinguish them from the numerical values
            that are computed later in the function

            Prefixes and suffixes:
            ob - observation
            ac - action
            _no - this tensor should have shape (batch self.size /n/, observation dim)
            _na - this tensor should have shape (batch self.size /n/, action dim)
            _n  - this tensor should have shape (batch self.size /n/)

            Note: batch self.size /n/ is defined at runtime, and until then, the shape for that axis
            is None

            ----------------------------------------------------------------------------------
            loss: a function of self.sy_logprob_n and self.sy_adv_n that we will differentiate
                to get the policy gradient.
        """
        self.sy_ob_no, self.sy_ac_na, self.sy_adv_n = self.define_placeholders()


        # The policy takes in an observation and produces a distribution over the action space
        self.policy_parameters = self.policy_forward_pass(self.sy_ob_no)

        # We can sample actions from this action distribution.
        # This will be called in Agent.sample_trajectory() where we generate a rollout.
        self.sy_sampled_ac = self.sample_action(self.policy_parameters)

        # We can also compute the logprob of the actions that were actually taken by the policy
        # This is used in the loss function.
        self.sy_logprob_n = self.get_log_prob(self.policy_parameters, self.sy_ac_na)

        #========================================================================================#
        #                           ----------PROBLEM 2----------
        # Loss Function and Training Operation
        #========================================================================================#
        weighted_neg_like = tf.multiply(-self.sy_logprob_n, self.sy_adv_n)
        self.loss =  tf.reduce_mean(weighted_neg_like)   # MY LOSS HERE
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        #========================================================================================#
        #                           ----------PROBLEM 6----------
        # Optional Baseline
        #
        # Define placeholders for targets, a loss function and an update op for fitting a
        # neural network baseline. These will be used to fit the neural network baseline.
        #========================================================================================#
        if self.nn_baseline:

            self.baseline_prediction = tf.squeeze(build_mlp(
                                    self.sy_ob_no,
                                    1,
                                    "nn_baseline",
                                    n_layers=self.n_layers,
                                    size=self.size))
            # YOUR_CODE_HERE

            # This network takes in the observation and predicts the expected return
            self.sy_target_n = tf.placeholder(shape=[None], name="targ", dtype=tf.float32)

            # I will train this to match the advtanges!
            self.baseline_loss = tf.losses.mean_squared_error(labels=self.sy_target_n, predictions =self.baseline_prediction)

            self.baseline_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.baseline_loss)

    def sample_trajectories(self, itr, env):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and self.animate)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        return paths, timesteps_this_batch

    def sample_trajectory(self, env, animate_this_episode):
        ob = env.reset()
        obs, acs, rewards = [], [], []
        steps = 0
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(.1)
            obs.append(ob)
            #====================================================================================#
            #                           ----------PROBLEM 3----------
            #====================================================================================#

            # Use the tensorflow session tocall self.sy_sampled_ac to sample an action given an observation from the env
            # print("ob is:", ob)
            ac = self.sess.run([self.sy_sampled_ac], feed_dict={self.sy_ob_no: ob.reshape(-1, self.ob_dim)})
            # print("AC IS, IN SIM: ", ac[0][0])

            # Hi person, sorry you are looking at my messy code. These if elses
            #   were made for debugging, and I have more important things to do
            #   than submit clean HW code
            if self.discrete:
                acs.append(ac[0][0][0])
            else:
                acs.append(ac[0][0])
            if self.ac_dim ==1:
                ob, rew, done, _ = env.step(ac[0][0][0])
            else:
                if self.discrete:
                    ob, rew, done, _ = env.step(np.squeeze(ac[0][0][0]))
                else:
                    ob, rew, done, _ = env.step(np.squeeze(ac[0][0]))
            rewards.append(rew)
            steps += 1
            if done or steps > self.max_path_length:
                break

        # print(acs)
        path = {"observation" : np.array(obs, dtype=np.float32),
                "reward" : np.array(rewards, dtype=np.float32),
                "action" : np.array(acs, dtype=np.float32)}
        return path

    #====================================================================================#
    #                           ----------PROBLEM 3----------
    #====================================================================================#
    def sum_of_rewards(self, re_n):
        """
            Monte Carlo estimation of the Q function.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                re_n: length: num_paths. Each element in re_n is a numpy array
                    containing the rewards for the particular path

            returns:
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
                    whose length is the sum of the lengths of the paths

            ----------------------------------------------------------------------------------

            Your code should construct numpy arrays for Q-values which will be used to compute
            advantages (which will in turn be fed to the placeholder you defined in
            Agent.define_placeholders).

            Recall that the expression for the policy gradient PG is

                  PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]

            where

                  tau=(s_0, a_0, ...) is a trajectory,
                  Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
                  and b_t is a baseline which may depend on s_t.

            You will write code for two cases, controlled by the flag 'reward_to_go':

              Case 1: trajectory-based PG

                  (reward_to_go = False)

                  Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over
                  entire trajectory (regardless of which time step the Q-value should be for).

                  For this case, the policy gradient estimator is

                      E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]

                  where

                      Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.

                  Thus, you should compute

                      Q_t = Ret(tau)

              Case 2: reward-to-go PG

                  (reward_to_go = True)

                  Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
                  from time step t. Thus, you should compute

                      Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}


            Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
            like the 'ob_no' and 'ac_na' above.
        """
        # YOUR_CODE_HERE
        gamma = self.gamma  # grabbed for variable name
        sum_of_path_lengths = 0
        for re in re_n:
            sum_of_path_lengths += len(re)

        if self.reward_to_go:

            q_n = np.zeros([sum_of_path_lengths], dtype=np.float32)
            # For each trajectory, compute the reward at each element
            # Then, concatenate the arrays of all the rewards on end
            # Can do this by putting them all in an array of length sum_of_path_lengths
            ind = 0
            # iterate through each set of rewards
            for re in re_n:
                len_re = len(re)
                for g in range(len_re):
                    rew_to_goes = [r*(gamma**i) for i,r in enumerate(re[g:])]
                    q_n[ind] = np.sum(rew_to_goes)
                    ind+= 1

            # Reset to mean 0 variance of 1
            # q_n -= np.mean(q_n)
            # q_n /= np.std(q_n)

        else:

            # Need to expand all the trajectories in re_n
            q_n = np.zeros([sum_of_path_lengths], dtype=np.float32)
            # For each trajectory, compute the reward across the entire trajectory
            # Then, set the reward for each element as the traj_reward

            ind = 0
            # iterate through each set of rewards
            for re in re_n:
                len_re = len(re)
                # print("Length of ep: ", len_re)
                traj_reward=0
                for g,r in enumerate(re):
                    traj_reward += r*(gamma**g)
                # print(traj_reward)
                q_n[ind:ind+len_re] = traj_reward

                ind+= len_re

            # Reset to mean 0 variance of 1
            # q_n -= np.mean(q_n)
            # q_n /= np.std(q_n)
        # print("Q_N", q_n)
        return q_n

    def compute_advantage(self, ob_no, q_n):
        """
            Computes advantages by (possibly) subtracting a baseline from the estimated Q values

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
                    whose length is the sum of the lengths of the paths

            returns:
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                    advantages whose length is the sum of the lengths of the paths
        """
        #====================================================================================#
        #                           ----------PROBLEM 6----------
        # Computing Baselines
        #====================================================================================#
        if self.nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current batch of Q-values. (Goes with Hint
            # #bl2 in Agent.update_parameters.

            b_n  = self.sess.run([self.baseline_prediction], feed_dict={self.sy_ob_no: ob_no})
            # print("q_n is:", q_n)
            # Need to normalize b_n to match the statistics of q_n
            # q_mean, q_var = tf.nn.moments(tf.convert_to_tensor(q_n), axes=[0],keep_dims = True)
            # q_n -= q_mean
            # q_n /= tf.sqrt(q_var)
            # # print("q_n is norm:", q_n)
            # # print(np.squeeze(b_n))
            # b_n = np.array(b_n)
            b_n = np.squeeze(b_n)
            # # print(b_n[0])
            # # print(np.shape(b_n[0]))
            # # print(tf.convert_to_tensor(b_n[0]))
            # b_mean, b_var = tf.nn.moments(tf.convert_to_tensor(b_n), axes=[0],keep_dims = True)
            # b_n -= b_mean
            # b_n /= tf.sqrt(b_var)
            b_n = ( (b_n - np.mean(b_n)) / np.std(b_n) )* np.std(q_n) + np.mean(q_n)

            adv_n = q_n - b_n

        else:
            adv_n = q_n.copy()
        return adv_n

    def estimate_return(self, ob_no, re_n):
        """
            Estimates the returns over a set of trajectories.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                re_n: length: num_paths. Each element in re_n is a numpy array
                    containing the rewards for the particular path

            returns:
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
                    whose length is the sum of the lengths of the paths
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                    advantages whose length is the sum of the lengths of the paths
        """
        q_n = self.sum_of_rewards(re_n)
        adv_n = self.compute_advantage(ob_no, q_n)
        #====================================================================================#
        #                           ----------PROBLEM 3----------
        # Advantage Normalization
        #====================================================================================#
        if self.normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.

            mean, var = tf.nn.moments(tf.convert_to_tensor(adv_n), axes=[0],keep_dims = True)
            adv_n -= mean
            adv_n /= tf.sqrt(var)
            adv_n = adv_n.eval()

        # print("ESTIMATED RETURN")
        # print(q_n)
        # print(adv_n)
        return q_n, adv_n

    def update_parameters(self, ob_no, ac_na, q_n, adv_n):
        """
            Update the parameters of the policy and (possibly) the neural network baseline,
            which is trained to approximate the value function.

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: shape: (sum_of_path_lengths).
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
                    whose length is the sum of the lengths of the paths
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                    advantages whose length is the sum of the lengths of the paths

            returns:
                nothing

        """
        #====================================================================================#
        #                           ----------PROBLEM 6----------
        # Optimizing Neural Network Baseline
        #====================================================================================#
        if self.nn_baseline:
            # If a neural network baseline is used, set up the targets and the inputs for the
            # baseline.
            #
            # Fit it to the current batch in order to use for the next iteration. Use the
            # baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the
            # targets to have mean zero and std=1. (Goes with Hint #bl1 in
            # Agent.compute_advantage.)

            # YOUR_CODE_HERE
            target_n = q_n.copy()
            t_mean, t_var = tf.nn.moments(tf.convert_to_tensor(target_n), axes=[0],keep_dims = True)
            target_n -= t_mean
            target_n /= tf.sqrt(t_var)
            target_n = target_n.eval()

            # loss_pre = self.sess.run([self.baseline_loss], feed_dict={self.sy_ob_no: ob_no, self.sy_ac_na: ac_na.reshape(-1), self.sy_target_n: target_n })
            # print("LOSS PRE: ", loss_pre)
            if self.discrete:
                grad = self.sess.run([self.baseline_update_op], feed_dict={self.sy_ob_no: ob_no, self.sy_ac_na: ac_na.reshape(-1), self.sy_target_n: target_n })
            else:
                grad = self.sess.run([self.baseline_update_op], feed_dict={self.sy_ob_no: ob_no, self.sy_ac_na: ac_na.reshape(-1,self.ac_dim), self.sy_target_n: target_n })
            # loss_post = self.sess.run([self.baseline_loss], feed_dict={self.sy_ob_no: ob_no, self.sy_ac_na: ac_na.reshape(-1), self.sy_target_n: target_n })
            # print("LOSS POST: ", loss_post)

        #====================================================================================#
        #                           ----------PROBLEM 3----------
        # Performing the Policy Update
        #====================================================================================#

        # Call the update operation necessary to perform the policy gradient update based on
        # the current batch of rollouts.
        #
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below.

        # YOUR_CODE_HERE
        # print("PARAM UPDATE: \n")
        # print(ob_no)

        # print("Compiled Actions:", ac_na)
        # print(self.ac_dim)
        # print(adv_n)
        # loss_pre = self.sess.run([self.loss], feed_dict={self.sy_ob_no: ob_no, self.sy_ac_na: ac_na.reshape(-1), self.sy_adv_n: adv_n })
        # print("LOSS PRE: ", loss_pre)
        if self.discrete:
            grad = self.sess.run([self.update_op], feed_dict={self.sy_ob_no: ob_no, self.sy_ac_na: ac_na.reshape(-1), self.sy_adv_n: adv_n })
        else:
            grad = self.sess.run([self.update_op], feed_dict={self.sy_ob_no: ob_no, self.sy_ac_na: ac_na.reshape(-1,self.ac_dim), self.sy_adv_n: adv_n })
        # loss_post = self.sess.run([self.loss], feed_dict={self.sy_ob_no: ob_no, self.sy_ac_na: ac_na, self.sy_adv_n: adv_n })
        # print("LOSS POST: ", loss_post)

def train_PG(
        exp_name,
        env_name,
        n_iter,
        gamma,
        min_timesteps_per_batch,
        max_path_length,
        learning_rate,
        reward_to_go,
        animate,
        logdir,
        normalize_advantages,
        nn_baseline,
        seed,
        n_layers,
        size):

    start = time.time()

    #========================================================================================#
    # Set Up Logger
    #========================================================================================#
    setup_logger(logdir, locals())

    #========================================================================================#
    # Set Up Env
    #========================================================================================#

    # Make the gym environment
    env = gym.make(env_name)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    # Is this env continuous, or self.discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
    # REFERENCE
    print('for reference!')
    print("action dimension is! ", ac_dim)
    print("is it discrete? ", discrete)
    #========================================================================================#
    # Initialize Agent
    #========================================================================================#
    computation_graph_args = {
        'n_layers': n_layers,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'discrete': discrete,
        'size': size,
        'learning_rate': learning_rate,
        }

    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }

    estimate_return_args = {
        'gamma': gamma,
        'reward_to_go': reward_to_go,
        'nn_baseline': nn_baseline,
        'normalize_advantages': normalize_advantages,
    }

    agent = Agent(computation_graph_args, sample_trajectory_args, estimate_return_args)

    # build computation graph
    agent.build_computation_graph()

    # tensorflow: config, session, variable initialization
    agent.init_tf_sess()

    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0
    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)
        paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        re_n = [path["reward"] for path in paths]

        q_n, adv_n = agent.estimate_return(ob_no, re_n)
        agent.update_parameters(ob_no, ac_na, q_n, adv_n)

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    processes = []

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)
        # print(args.dont_normalize_advantages)

        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline,
                seed=seed,
                n_layers=args.n_layers,
                size=args.size
                )
        # # Awkward hacky process runs, because Tensorflow does not like
        # # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        processes.append(p)
        # if you comment in the line below, then the loop will block
        # until this process finishes
        # p.join()

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
