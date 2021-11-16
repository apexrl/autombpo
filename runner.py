import tensorflow as tf
from softlearning.environments.utils import get_environment_from_params
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import get_policy_from_variant, get_policy
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant
from softlearning.misc.utils import set_seed, initialize_tf_variables
from controller import Controller
import tensorflow.contrib.slim as slim

import static
import time
import os
import random

class ExperimentRunner:
    def __init__(self, variant):
        set_seed(random.randint(0,1e9))
        # self.experiment_id = variant['algorithm_params']['exp_name']
        # self.local_dir = os.path.join(variant['algorithm_params']['log_dir'], variant['algorithm_params']['domain'])
        self._mode = variant['run_params']['mode']
        self._model_path = variant['run_params']['model_path']
        self._exp_id = variant['run_params']['exp_id']
        self._epsilon = variant['run_params']['epsilon']
        self.variant = variant

        hyperparameters_set = self.variant['controller_params']['hyperparameters_set']
        controllers_init = self.variant['controller_params']['controllers_init']

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True

        self._controller_graphs, self._controller_sessions, self._controllers = [],[],[]

        for i in range(len(hyperparameters_set)):
            hyperparameters = hyperparameters_set[i]
            if(i == 0):
                state_dim = 4 + len(hyperparameters)
            else:
                state_dim = 4 + len(hyperparameters_set[0]) + len(hyperparameters)
            controller_graph = tf.Graph()
            controller_session = tf.InteractiveSession(config=config, graph=controller_graph)
            action_dim = len(hyperparameters)
            action_space = []
            for hyperparameeter in hyperparameters:
                if (hyperparameeter == 'model'):
                    action_space.append(2)
                else:
                    action_space.append(3)
            controller = Controller(state_dim=state_dim, action_dim=action_dim, action_space=action_space,
                                    hyperparameters=hyperparameters,init = controllers_init[i],
                                    graph=controller_graph,session=controller_session)
            self._controller_graphs.append(controller_graph)
            self._controller_sessions.append(controller_session)
            self._controllers.append(controller)

        self._mbrl_graph = tf.Graph()
        self._mbrl_session = tf.InteractiveSession(config=config, graph=self._mbrl_graph)


        self.log_path = './log/%s/%d' % (self.variant['algorithm_params']['domain'], self._exp_id/10)
        if (not os.path.exists(self.log_path)):
            os.makedirs(self.log_path)
        self.log_name = '/%d_%d.log' % (time.time(),self._exp_id)
        self.log_file = self.log_path + self.log_name

        self.train_generator = None

    def build(self):
        with self._mbrl_graph.as_default():
            environment_params = self.variant['environment_params']
            training_environment = self.training_environment = (
                get_environment_from_params(environment_params['training']))
            evaluation_environment = self.evaluation_environment = (
                get_environment_from_params(environment_params['evaluation'])
                if 'evaluation' in environment_params
                else training_environment)

            replay_pool = self.replay_pool = (get_replay_pool_from_variant(self.variant, training_environment))
            sampler = self.sampler = get_sampler_from_variant(self.variant)
            Qs = self.Qs = get_Q_function_from_variant(self.variant, training_environment)
            policy = self.policy = get_policy_from_variant(self.variant, training_environment, Qs)
            initial_exploration_policy = self.initial_exploration_policy = (
                get_policy('UniformPolicy', training_environment))

            #### get termination function
            self._domain = environment_params['training']['domain']
            static_fns = static[self._domain.lower()]
            ####

            self.algorithm = get_algorithm_from_variant(
                variant=self.variant,
                training_environment=training_environment,
                evaluation_environment=evaluation_environment,
                policy=policy,
                initial_exploration_policy=initial_exploration_policy,
                Qs=Qs,
                pool=replay_pool,
                static_fns=static_fns,
                sampler=sampler,
                session=self._mbrl_session,
                controllers=self._controllers,
                mode = self._mode,
                exp_id = self._exp_id,
                epsilon = self._epsilon,
                domain = self._domain,
                log_file=self.log_file)

            self._init_mbrl = tf.global_variables_initializer()
        initialize_tf_variables(self._mbrl_session, only_uninitialized=True)


    def train(self):
        self.build()
        if(self._mode == 'eval'):
            for i in range(len(self._controllers)):
                path = self._model_path + '/' + self._domain + '/controller' + str(i) + '/best'
                self._controllers[i].restore(path = path)
        else:
            for i in range(len(self._controllers)-1):
                path = self._model_path + '/' + self._domain + '/controller' + str(i) + '/best'
                self._controllers[i].restore(path = path)
            path = self._model_path + '/' + self._domain + '/controller' + str(len(self._controllers)-1) + '/current'
            self._controllers[-1].restore(path=path)
        self.algorithm.train()

