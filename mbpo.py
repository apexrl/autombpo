import math
from collections import OrderedDict
from numbers import Number
from itertools import count
import os
import random
import copy

import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util

from softlearning.algorithms.rl_algorithm import RLAlgorithm
from softlearning.replay_pools.simple_replay_pool import SimpleReplayPool

from models.constructor import construct_model, format_samples_for_training
from models.fake_env import FakeEnv


def td_target(reward, discount, next_value):
    return reward + discount * next_value


class MBPO(RLAlgorithm):

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policy,
            Qs,
            pool,
            static_fns,
            log_file=None,
            plotter=None,
            tf_summaries=False,

            lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,
            store_extra_policy_info=False,

            deterministic=False,
            model_rollout_freq=250,
            controller_tau=125,
            num_networks=7,
            num_elites=5,
            model_retain_epochs=20,
            rollout_batch_size=100e3,
            real_ratio=0.1,
            rollout_schedule=[20, 100, 1, 1],
            hidden_dim=200,
            max_model_t=None,
            controllers = None,
            mode = 'eval',
            exp_id = 0,
            epsilon = 0,
            domain = 'Ant',
            real_ratio_c = 1.2,
            model_train_penalty = 0.1,
            Reward_scale = 300,
            model_loss_scale = [0, 1],
            value_loss_scale = [0, 1],
            policy_error_scale = [0, 1],
            performance_scale = [0, 1],
            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
        """

        super(MBPO, self).__init__(**kwargs)

        obs_dim = np.prod(training_environment.observation_space.shape)
        act_dim = np.prod(training_environment.action_space.shape)
        self._model = construct_model(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim,
                                      num_networks=num_networks, num_elites=num_elites,session=self._session)
        self._static_fns = static_fns
        self.fake_env = FakeEnv(self._model, self._static_fns)

        self._rollout_schedule = rollout_schedule
        self._max_model_t = max_model_t

        self._model_retain_epochs = model_retain_epochs

        self._model_rollout_freq = model_rollout_freq
        self._controller_tau = controller_tau
        self._rollout_batch_size = int(rollout_batch_size)
        self._deterministic = deterministic
        self._real_ratio = real_ratio

        self._controllers = controllers
        self._mode = mode
        self._exp_id = exp_id
        self._epsilon = epsilon
        self._domain = domain
        self._real_ratio_c = real_ratio_c
        self._model_train_penalty = model_train_penalty
        self._Reward_scale = Reward_scale
        self._model_loss_scale = model_loss_scale
        self._value_loss_scale = value_loss_scale
        self._policy_error_scale = policy_error_scale
        self._performance_scale = performance_scale

        self._log_dir = os.getcwd()

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy

        self._Qs = Qs
        self._Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in Qs)

        self._pool = pool
        self._plotter = plotter
        self._tf_summaries = tf_summaries

        self._policy_lr = lr
        self._Q_lr = lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            -np.prod(self._training_environment.action_space.shape)
            if target_entropy == 'auto'
            else target_entropy)
        print('[ MBPO ] Target entropy: {}'.format(self._target_entropy))

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._reparameterize = reparameterize
        self._store_extra_policy_info = store_extra_policy_info

        observation_shape = self._training_environment.active_observation_shape
        action_shape = self._training_environment.action_space.shape

        assert len(observation_shape) == 1, observation_shape
        self._observation_shape = observation_shape
        assert len(action_shape) == 1, action_shape
        self._action_shape = action_shape
        self.log_file = log_file

        self._build()

    def _build(self):
        self._training_ops = {}

        self._init_global_step()
        self._init_placeholders()
        self._init_actor_update()
        self._init_critic_update()

    def _train(self):

        training_environment = self._training_environment
        evaluation_environment = self._evaluation_environment
        policy = self._policy
        pool = self._pool
        self._model_metrics = {}

        if not self._training_started:
            self._init_training()

            self._initial_exploration_hook(
                training_environment, self._initial_exploration_policy, pool)

        self.sampler.initialize(training_environment, policy, pool)

        self._training_before_hook()

        self._schedule_rollout_length = False
        self._schedule_policy_training = False
        for i in range(len(self._controllers)):
            if('rollout' in self._controllers[i]._hyperparameters):
                self._schedule_rollout_length = True
            if ('policy' in self._controllers[i]._hyperparameters):
                self._n_train_repeat = 10
        self._rollout_length = 1
        self._baseline = np.load('baselines/' + self._domain + '/baseline.npy')
        #hyper-MDP trajectory
        self._States, self._Actions, self._Logps, self._Rewards = [], [], [], []

        #set model loss
        model_train_metrics = self._train_model(batch_size=256, max_epochs=None, holdout_ratio=0.2,
                                                max_t=self._max_model_t)
        self._model_loss = model_train_metrics['val_loss']
        self._model_metrics.update(model_train_metrics)

        # set evaluation metrics
        evaluation_returns = []
        evaluation_paths = self._evaluation_paths(
            policy, evaluation_environment)
        evaluation_metrics = self._evaluate_rollouts(
            evaluation_paths, evaluation_environment)
        self._evaluation_performance = (evaluation_metrics['return-average'])
        for self._epoch in range(self._epoch, self._n_epochs):

            self._epoch_before_hook()
            self._model_train_repeat = 0

            start_samples = self.sampler._total_samples
            print("\033[0;31m%s%d\033[0m" % ('epoch: ', self._epoch))
            print('[ True Env Buffer Size ]', pool.size)
            for _ in count():
                samples_now = self.sampler._total_samples
                self._timestep = samples_now - start_samples

                if samples_now >= start_samples + self._epoch_length and self.ready_to_train:
                    break

                self._timestep_before_hook()

                if self._timestep % (self._controller_tau) == 0:

                    states = self._construct_states(controllers=self._controllers)
                    self._States.append(states[-1])
                    self._Rewards.append(0)

                    for k in range(len(self._controllers)):
                        #epsilon greedy
                        if(random.random() < self._epsilon):
                            action_space = self._controllers[k]._action_space
                            action = np.zeros(shape=(len(action_space)))
                            for j in range(len(action_space)):
                                action[j] = np.random.randint(0,action_space[j])
                            logp = self._controllers[k].get_logp(states[k],action)
                        else:
                            action,logp = self._controllers[k].get_action(states[k])
                        self._response(action = action, hyperparameters = self._controllers[k]._hyperparameters)
                    self._Actions.append(action)
                    self._Logps.append(logp)


                    f_log = open(self.log_file, 'a')
                    f_log.write('Set rollout length: %d\n' % self._rollout_length)
                    f_log.write('Set train repeat: %d\n' % self._n_train_repeat)
                    f_log.write('Set real ratio: %f\n' % self._real_ratio)
                    f_log.close()


                if self._timestep % self._model_rollout_freq == 0:
                    if(not self._schedule_rollout_length):
                        self._set_rollout_length()
                    self._reallocate_model_pool()
                    model_rollout_metrics = self._rollout_model(rollout_batch_size=self._rollout_batch_size,
                                                                deterministic=self._deterministic)
                    self._model_metrics.update(model_rollout_metrics)

                self._do_sampling(timestep=self._total_timestep)

                if self.ready_to_train:
                    self._do_training_repeats(timestep=self._total_timestep)

                self._timestep_after_hook()

            training_paths = self.sampler.get_last_n_paths(
                math.ceil(self._epoch_length / self.sampler._max_path_length))
            evaluation_paths = self._evaluation_paths(
                policy, evaluation_environment)

            training_metrics = self._evaluate_rollouts(
                training_paths, training_environment)
            if evaluation_paths:
                evaluation_metrics = self._evaluate_rollouts(
                    evaluation_paths, evaluation_environment)
                evaluation_returns.append(evaluation_metrics['return-average'])
                self._evaluation_performance = (evaluation_metrics['return-average'])
            else:
                evaluation_metrics = {}

            #Construct hyper-MDP reward
            Reward = (evaluation_returns[-1] - self._baseline[self._epoch]) / self._Reward_scale
            self._Rewards[-1] += Reward


            self._epoch_after_hook(training_paths)

            sampler_diagnostics = self.sampler.get_diagnostics()

            diagnostics = self.get_diagnostics(
                iteration=self._total_timestep,
                batch=self._evaluation_batch(),
                training_paths=training_paths,
                evaluation_paths=evaluation_paths)

            diagnostics.update(OrderedDict((
                *(
                    (f'evaluation/{key}', evaluation_metrics[key])
                    for key in sorted(evaluation_metrics.keys())
                ),
                *(
                    (f'training/{key}', training_metrics[key])
                    for key in sorted(training_metrics.keys())
                ),
                *(
                    (f'sampler/{key}', sampler_diagnostics[key])
                    for key in sorted(sampler_diagnostics.keys())
                ),
                *(
                    (f'model/{key}', self._model_metrics[key])
                    for key in sorted(self._model_metrics.keys())
                ),
                ('epoch', self._epoch),
                ('timestep', self._timestep),
                ('timesteps_total', self._total_timestep),
                ('train-steps', self._num_train_steps),
            )))

            if self._eval_render_mode is not None and hasattr(
                    evaluation_environment, 'render_rollouts'):
                training_environment.render_rollouts(evaluation_paths)
            print(diagnostics)
            f_log = open(self.log_file, 'a')
            f_log.write('epoch: %d\n' % self._epoch)
            f_log.write('Model training repeat: %d\n' % self._model_train_repeat)
            f_log.write('total time steps: %d\n' % self._total_timestep)
            f_log.write('evaluation return: %f\n' % evaluation_metrics['return-average'])
            f_log.write('Qfunction loss: %f\n' % diagnostics['Q_loss'])
            f_log.close()


        #calculate the adv and store data to the buffer
        if(self._mode == 'train'):
            advs = np.zeros(shape=(len(self._Rewards), 1))
            tmp_adv = 0
            length = len(self._Rewards)
            for i in range(length):
                tmp_adv = tmp_adv + self._Rewards[length - 1 - i]
                advs[length - 1 - i][0] = tmp_adv

            np.save('./buffer/states_%d.npy' %(self._exp_id), np.array(self._States))
            np.save('./buffer/actions_%d.npy' % (self._exp_id), np.array(self._Actions))
            np.save('./buffer/advs_%d.npy' % (self._exp_id), np.array(advs))
            np.save('./buffer/logps_%d.npy' % (self._exp_id), np.array(self._Logps))
            np.save('./buffer/mbrl_returns_%d.npy' % (self._exp_id), np.array(evaluation_returns))

        self.sampler.terminate()

        self._training_after_hook()

        # self._training_progress.close()

    def _construct_states(self, controllers):
        # construct training percentile
        self._training_percentile = self._epoch / self._n_epochs + self._timestep / (
                self._epoch_length * self._n_epochs)

        # construct Q_loss
        if not hasattr(self, '_model_pool'):
            batch = self._env_batch()
        else:
            batch = self._training_batch(batch_size=int(1e5))
        feed_dict = self._get_feed_dict(self._total_timestep, batch)
        self._value_loss = np.mean(self._session.run(
            self._Q_losses, feed_dict))

        # construct policy shift error
        if (self._timestep == 0):
            batch = self._pool.last_n_batch(last_n=5 * self._epoch_length)
            obs = batch['observations']
            act = batch['actions']
            old_log_pis = np.clip(np.nan_to_num(batch['log_pis']), -1e2, 1e2)
            log_pis = np.clip(np.nan_to_num(self._policy.log_pis_np([obs], act)), -1e2, 1e2)
            old_pis = np.exp(old_log_pis)
            pis = np.exp(log_pis)
            self._policy_shift_error = np.mean(np.minimum(np.abs(old_pis - pis), 100))

        states = []
        base_state = [self._training_percentile,
                          (self._model_loss - self._model_loss_scale[0]) / (
                                      self._model_loss_scale[1] - self._model_loss_scale[0]),
                          (self._value_loss - self._value_loss_scale[0]) / (
                                      self._value_loss_scale[1] - self._value_loss_scale[0]),
                          (self._policy_shift_error - self._policy_error_scale[0]) / (
                                      self._policy_error_scale[1] - self._policy_error_scale[0]),
                          (self._evaluation_performance - self._performance_scale[0]) / (
                                      self._performance_scale[1] - self._performance_scale[0])]
        if ('policy' in controllers[0]._hyperparameters):
            base_state.append(self._n_train_repeat / 20)
        if ('ratio' in controllers[0]._hyperparameters):
            base_state.append(self._real_ratio)
        if ('rollout' in controllers[0]._hyperparameters):
            min_epoch, max_epoch, min_length, max_length = self._rollout_schedule
            base_state.append(self._rollout_length / max_length)
        states.append(np.clip(base_state, -1, 1))
        for i in range(1,len(controllers)):
            temp_state = copy.deepcopy(base_state)
            if('policy' in controllers[i]._hyperparameters):
                temp_state.append(self._n_train_repeat/20)
            if ('ratio' in controllers[i]._hyperparameters):
                temp_state.append(self._real_ratio)
            if('rollout' in controllers[i]._hyperparameters):
                min_epoch, max_epoch, min_length, max_length = self._rollout_schedule
                temp_state.append(self._rollout_length/max_length)
            temp_state = np.clip(temp_state, -1, 1)
            states.append(temp_state)
        return states

    def _response(self, action, hyperparameters):
        for j in range(len(hyperparameters)):
            if(hyperparameters[j] == 'model'):
                #Give a penalty if the controller choose to train the model
                self._Rewards[-1] += self._model_train_penalty*(action[j]-0.5)
                if (action[j] == 0):
                    model_train_metrics = self._train_model(batch_size=256, max_epochs=None, holdout_ratio=0.2,
                                                            max_t=self._max_model_t)
                    self._model_loss = model_train_metrics['val_loss']
                    self._model_metrics.update(model_train_metrics)
                    self._model_train_repeat += 1
                    f_log = open(self.log_file, 'a')
                    f_log.write('model loss: %f\n' % model_train_metrics['val_loss'])
                    f_log.close()

            elif(hyperparameters[j] == 'ratio'):
                self._real_ratio = self._real_ratio * (self._real_ratio_c ** int(action[j] - 1))
                self._real_ratio = min(1.0, self._real_ratio)

            elif(hyperparameters[j] == 'policy'):
                self._n_train_repeat = self._n_train_repeat + int(action[j]) - 1
                self._n_train_repeat = max(1, min(self._n_train_repeat, 20))

            elif (self._timestep == 0 and hyperparameters[j] == 'rollout'):
                min_epoch, max_epoch, min_length, max_length = self._rollout_schedule
                self._rollout_length = self._rollout_length + int(action[j]) - 1
                self._rollout_length = max(1, min(self._rollout_length, max_length))




    def train(self, *args, **kwargs):
        return self._train(*args, **kwargs)

    # def _log_policy(self):
    #     save_path = os.path.join(self._log_dir, 'models')
    #     filesystem.mkdir(save_path)
    #     weights = self._policy.get_weights()
    #     data = {'policy_weights': weights}
    #     full_path = os.path.join(save_path, 'policy_{}.pkl'.format(self._total_timestep))
    #     print('Saving policy to: {}'.format(full_path))
    #     pickle.dump(data, open(full_path, 'wb'))

    # def _log_model(self):
    #     save_path = os.path.join(self._log_dir, 'models')
    #     filesystem.mkdir(save_path)
    #     print('Saving model to: {}'.format(save_path))
    #     self._model.save(save_path, self._total_timestep)

    def _set_rollout_length(self):
        min_epoch, max_epoch, min_length, max_length = self._rollout_schedule
        if self._epoch <= min_epoch:
            y = min_length
        else:
            dx = (self._epoch - min_epoch) / (max_epoch - min_epoch)
            dx = min(dx, 1)
            y = dx * (max_length - min_length) + min_length

        self._rollout_length = int(y)
        print('[ Model Length ] Epoch: {} (min: {}, max: {}) | Length: {} (min: {} , max: {})'.format(
            self._epoch, min_epoch, max_epoch, self._rollout_length, min_length, max_length
        ))

    def _reallocate_model_pool(self):
        obs_space = self._pool._observation_space
        act_space = self._pool._action_space

        rollouts_per_epoch = self._rollout_batch_size * self._epoch_length / self._model_rollout_freq
        model_steps_per_epoch = int(self._rollout_length * rollouts_per_epoch)
        new_pool_size = self._model_retain_epochs * model_steps_per_epoch

        if not hasattr(self, '_model_pool'):
            print('[ Allocate Model Pool ] Initializing new model pool with size {:.2e}'.format(
                new_pool_size
            ))
            self._model_pool = SimpleReplayPool(obs_space, act_space, new_pool_size)

        elif self._model_pool._max_size != new_pool_size:
            print('[ Reallocate Model Pool ] Updating model pool | {:.2e} --> {:.2e}'.format(
                self._model_pool._max_size, new_pool_size
            ))
            samples = self._model_pool.return_all_samples()
            new_pool = SimpleReplayPool(obs_space, act_space, new_pool_size)
            new_pool.add_samples(samples)
            # assert self._model_pool.size == new_pool.size
            self._model_pool = new_pool

    def _train_model(self, **kwargs):
        env_samples = self._pool.return_all_samples()
        train_inputs, train_outputs = format_samples_for_training(env_samples)
        model_metrics = self._model.train(train_inputs, train_outputs, **kwargs)
        return model_metrics

    def _rollout_model(self, rollout_batch_size, **kwargs):
        print('[ Model Rollout ] Starting | Epoch: {} | Rollout length: {} | Batch size: {}'.format(
            self._epoch, self._rollout_length, rollout_batch_size
        ))
        batch = self.sampler.random_batch(rollout_batch_size)
        obs = batch['observations']
        steps_added = []
        for i in range(self._rollout_length):
            act = self._policy.actions_np(obs)

            next_obs, rew, term, info = self.fake_env.step(obs, act, **kwargs)
            steps_added.append(len(obs))

            samples = {'observations': obs, 'actions': act, 'next_observations': next_obs, 'rewards': rew,
                       'terminals': term}
            self._model_pool.add_samples(samples)

            nonterm_mask = ~term.squeeze(-1)
            if nonterm_mask.sum() == 0:
                print(
                    '[ Model Rollout ] Breaking early: {} | {} / {}'.format(i, nonterm_mask.sum(), nonterm_mask.shape))
                break

            obs = next_obs[nonterm_mask]

        mean_rollout_length = sum(steps_added) / rollout_batch_size
        rollout_stats = {'mean_rollout_length': mean_rollout_length}
        print('[ Model Rollout ] Added: {:.1e} | Model pool: {:.1e} (max {:.1e}) | Length: {} | Train rep: {}'.format(
            sum(steps_added), self._model_pool.size, self._model_pool._max_size, mean_rollout_length,
            self._n_train_repeat
        ))
        return rollout_stats

    # def _visualize_model(self, env, timestep):
    #     ## save env state
    #     state = env.unwrapped.state_vector()
    #     qpos_dim = len(env.unwrapped.sim.data.qpos)
    #     qpos = state[:qpos_dim]
    #     qvel = state[qpos_dim:]
    #
    #     print('[ Visualization ] Starting | Epoch {} | Log dir: {}\n'.format(self._epoch, self._log_dir))
    #     visualize_policy(env, self.fake_env, self._policy, self._writer, timestep)
    #     print('[ Visualization ] Done')
    #     ## set env state
    #     env.unwrapped.set_state(qpos, qvel)

    def _training_batch(self, batch_size=None):
        batch_size = batch_size or self.sampler._batch_size
        env_batch_size = int(batch_size * self._real_ratio)
        model_batch_size = batch_size - env_batch_size

        ## can sample from the env pool even if env_batch_size == 0
        env_batch = self._pool.random_batch(env_batch_size)

        if model_batch_size > 0:
            model_batch = self._model_pool.random_batch(model_batch_size)

            keys = model_batch.keys()
            batch = {k: np.concatenate((env_batch[k], model_batch[k]), axis=0) for k in keys}
        else:
            ## if real_ratio == 1.0, no model pool was ever allocated,
            ## so skip the model pool sampling
            batch = env_batch
        return batch

    def _env_batch(self, batch_size=None):
        batch_size = batch_size or self.sampler._batch_size
        return self._pool.random_batch(batch_size)


    def _init_global_step(self):
        self.global_step = training_util.get_or_create_global_step()
        self._training_ops.update({
            'increment_global_step': training_util._increment_global_step(1)
        })

    def _init_placeholders(self):
        """Create input placeholders for the SAC algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - next observation
            - action
            - reward
            - terminals
        """
        self._iteration_ph = tf.placeholder(
            tf.int64, shape=None, name='iteration')

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='observation',
        )

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='next_observation',
        )

        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._action_shape),
            name='actions',
        )

        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='rewards',
        )

        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='terminals',
        )

        if self._store_extra_policy_info:
            self._log_pis_ph = tf.placeholder(
                tf.float32,
                shape=(None, 1),
                name='log_pis',
            )
            self._raw_actions_ph = tf.placeholder(
                tf.float32,
                shape=(None, *self._action_shape),
                name='raw_actions',
            )

    def _get_Q_target(self):
        next_actions = self._policy.actions([self._next_observations_ph])
        next_log_pis = self._policy.log_pis(
            [self._next_observations_ph], next_actions)

        next_Qs_values = tuple(
            Q([self._next_observations_ph, next_actions])
            for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_value = min_next_Q - self._alpha * next_log_pis

        Q_target = td_target(
            reward=self._reward_scale * self._rewards_ph,
            discount=self._discount,
            next_value=(1 - self._terminals_ph) * next_value)

        return Q_target

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.
        """
        Q_target = tf.stop_gradient(self._get_Q_target())

        assert Q_target.shape.as_list() == [None, 1]

        Q_values = self._Q_values = tuple(
            Q([self._observations_ph, self._actions_ph])
            for Q in self._Qs)

        Q_losses = self._Q_losses = tuple(
            tf.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)

        self._Q_optimizers = tuple(
            tf.train.AdamOptimizer(
                learning_rate=self._Q_lr,
                name='{}_{}_optimizer'.format(Q._name, i)
            ) for i, Q in enumerate(self._Qs))
        Q_training_ops = tuple(
            tf.contrib.layers.optimize_loss(
                Q_loss,
                self.global_step,
                learning_rate=self._Q_lr,
                optimizer=Q_optimizer,
                variables=Q.trainable_variables,
                increment_global_step=False,
                summaries=((
                               "loss", "gradients", "gradient_norm", "global_gradient_norm"
                           ) if self._tf_summaries else ()))
            for i, (Q, Q_loss, Q_optimizer)
            in enumerate(zip(self._Qs, Q_losses, self._Q_optimizers)))

        self._training_ops.update({'Q': tf.group(Q_training_ops)})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.
        """

        actions = self._policy.actions([self._observations_ph])
        log_pis = self._policy.log_pis([self._observations_ph], actions)

        assert log_pis.shape.as_list() == [None, 1]

        log_alpha = self._log_alpha = tf.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

            self._alpha_optimizer = tf.train.AdamOptimizer(
                self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self._training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })

        self._alpha = alpha

        if self._action_prior == 'normal':
            policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self._action_shape),
                scale_diag=tf.ones(self._action_shape))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        Q_log_targets = tuple(
            Q([self._observations_ph, actions])
            for Q in self._Qs)
        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

        if self._reparameterize:
            policy_kl_losses = (
                    alpha * log_pis
                    - min_Q_log_target
                    - policy_prior_log_probs)
        else:
            raise NotImplementedError

        assert policy_kl_losses.shape.as_list() == [None, 1]

        policy_loss = tf.reduce_mean(policy_kl_losses)

        self._policy_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")
        policy_train_op = tf.contrib.layers.optimize_loss(
            policy_loss,
            self.global_step,
            learning_rate=self._policy_lr,
            optimizer=self._policy_optimizer,
            variables=self._policy.trainable_variables,
            increment_global_step=False,
            summaries=(
                "loss", "gradients", "gradient_norm", "global_gradient_norm"
            ) if self._tf_summaries else ())

        self._training_ops.update({'policy_train_op': policy_train_op})

    def _init_training(self):
        self._update_target(tau=1.0)

    def _update_target(self, tau=None):
        tau = tau or self._tau

        for Q, Q_target in zip(self._Qs, self._Q_targets):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""

        # self._training_progress.update()
        # self._training_progress.set_description()

        feed_dict = self._get_feed_dict(iteration, batch)

        self._session.run(self._training_ops, feed_dict)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._observations_ph: batch['observations'],
            self._actions_ph: batch['actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_ph: batch['rewards'],
            self._terminals_ph: batch['terminals'],
        }

        if self._store_extra_policy_info:
            feed_dict[self._log_pis_ph] = batch['log_pis']
            feed_dict[self._raw_actions_ph] = batch['raw_actions']

        if iteration is not None:
            feed_dict[self._iteration_ph] = iteration

        return feed_dict

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)

        (Q_values, Q_losses, alpha, global_step) = self._session.run(
            (self._Q_values,
             self._Q_losses,
             self._alpha,
             self.global_step),
            feed_dict)

        diagnostics = OrderedDict({
            'Q-avg': np.mean(Q_values),
            'Q-std': np.std(Q_values),
            'Q_loss': np.mean(Q_losses),
            'alpha': alpha,
        })

        policy_diagnostics = self._policy.get_diagnostics(
            batch['observations'])
        diagnostics.update({
            f'policy/{key}': value
            for key, value in policy_diagnostics.items()
        })

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_log_alpha': self._log_alpha,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables

    def reset(self,log_file=None):
        self._timestep = 0
        self._epoch = 0
        self._pool._pointer = 0
        self._pool._size = 0
        self._pool._samples_since_save = 0
        delattr(self, '_model_pool')
        self.log_file = log_file
        # self._model.reset()





