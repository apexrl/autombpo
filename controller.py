import numpy as np
import tensorflow as tf



class Buffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros((size, 1), dtype=np.float32)
        self.logp_buf = np.zeros((size, act_dim), dtype=np.float32)

        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self,obs,act,rew,logp):
        for i in range(np.shape(obs)[0]):
            self.obs_buf[self.ptr] = obs[i]
            self.act_buf[self.ptr] = act[i]
            self.adv_buf[self.ptr] = rew[i]
            self.logp_buf[self.ptr] = logp[i]
            self.ptr = (self.ptr + 1) % self.max_size
            if(self.size < self.max_size):
                self.size += 1

    def clear(self):
        self.ptr, self.size = 0, 0

    def get(self,batch_size = 64):
        batch_size = min(batch_size,self.size)
        inds = np.random.choice(self.size, batch_size, replace=False)
        return self.obs_buf[inds], self.act_buf[inds], self.adv_buf[inds], self.logp_buf[inds]

    def get_all(self):
        return self.get(batch_size=self.size)



class Controller:
    def __init__(
            self,
            state_dim = 6,
            action_dim = 2,
            action_space = [2,3],
            hyperparameters = ['model','ratio'],
            scope = 'controller',
            init = False,
            lr = 3e-4,
            epsilon = 0.2,
            buffer_size = 5000,
            graph = None,
            session=None,
    ):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._action_space = action_space
        self._hyperparameters = hyperparameters
        self._scope = scope
        self._init = init
        self._lr = lr
        self._epsilon = epsilon
        self._graph = graph
        self._session = session
        self._buffer = Buffer(self._state_dim,self._action_dim,buffer_size)


        self._build()

    def _build(self):
        with self._graph.as_default():
            self._init_placeholder()
            self._build_actor_net()

            self._saver = tf.train.Saver()

            self.init = tf.global_variables_initializer()
        self._session.run(self.init)


    def _init_placeholder(self):
        self._state_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._state_dim),
            name='observation',
        )

        self._actions_ph = tf.placeholder(
            tf.int32,
            shape=(None, self._action_dim),
            name='actions',
        )

        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='rewards',
        )

        self._old_logp_ph = tf.placeholder(
            tf.float32,
            shape = (None, self._action_dim),
            name='old_prob')

        self._advantage_ph = tf.placeholder(
            tf.float32,
            shape=(None,),
            name='advantage')




    def _build_actor_net(self):
        self._pi,self._logp,self._logp_pi,self._p_all = [],[],[],[]
        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            with tf.variable_scope('actor', reuse=tf.AUTO_REUSE):
                initializer = tf.random_normal_initializer(mean=0, stddev=0.04)
                for i in range(self._action_dim):
                    if(self._init):
                        hidden_layer1 = tf.layers.dense(self._state_ph, 256, tf.nn.tanh,kernel_initializer=initializer)
                        logits = tf.layers.dense(hidden_layer1, self._action_space[i],kernel_initializer=initializer)
                    else:
                        hidden_layer1 = tf.layers.dense(self._state_ph, 256, tf.nn.tanh)
                        logits = tf.layers.dense(hidden_layer1, self._action_space[i])
                    logp_all = tf.nn.log_softmax(logits)
                    p_all = tf.nn.softmax(logits)
                    pi = tf.multinomial(logits, 1)
                    logp = tf.reduce_sum(tf.one_hot(self._actions_ph[:, i], depth=self._action_space[i]) * logp_all, axis=1,
                                           keep_dims=True)
                    logp_pi = tf.reduce_sum(tf.one_hot(tf.squeeze(pi, axis=1), depth=self._action_space[i]) * logp_all, axis=1,
                                              keep_dims=True)
                    self._pi.append(pi)
                    self._logp.append(logp)
                    self._logp_pi.append(logp_pi)
                    self._p_all.append(p_all)
        self._actor_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=(self._scope + '/actor'))
        self._pi = tf.concat(self._pi, axis=1)
        self._logp = tf.concat(self._logp, axis=1)
        self._logp_pi = tf.concat(self._logp_pi, axis=1)
        self._p_all = tf.concat(self._p_all, axis=1)


        ratio = tf.exp((self._logp - self._old_logp_ph))
        surr = ratio * tf.expand_dims(self._advantage_ph,axis=1)

        self._actor_loss = -tf.reduce_mean(tf.reduce_sum(
            tf.minimum(surr, tf.clip_by_value(ratio, 1. - self._epsilon, 1. + self._epsilon) * tf.expand_dims(
                self._advantage_ph, axis=1)), axis = 1))


        self._actor_optimizer = tf.train.AdamOptimizer(self._lr).minimize(loss=self._actor_loss,
                                                                                 var_list=self._actor_params)


    def get_action(self,state):
        if(len(np.shape(state)) == 1):
            state = [state]
        action, logp_pi = self._session.run([self._pi, self._logp_pi], {self._state_ph: state})

        p_all = self._session.run(self._p_all, {self._state_ph: state})

        return action[0], logp_pi[0]

    def get_logp(self,state, action):
        logp = self._session.run(self._logp,{self._state_ph:[state],self._actions_ph:[action]})
        return logp[0]


    def store(self,states,actions,advs,old_logp):
        self._buffer.store(states,actions,advs,old_logp)

    def update(self,gradient_steps = 4,batch_size=64):
        if(self._buffer.size > 0):
            for i in range(gradient_steps):
                obs, act, adv, logp = self._buffer.get(batch_size=batch_size)
                adv = np.squeeze(adv, axis=1)
                feed_dict = {
                    self._state_ph: obs,
                    self._actions_ph: act,
                    self._advantage_ph: adv,
                    self._old_logp_ph: logp,
                }

                self._session.run(self._actor_optimizer, feed_dict)

    def clear(self):
        self._buffer.clear()

    def save(self, path = 'save/current'):
        self._saver.save(self._session, path)

    def restore(self, path = 'save/current'):
        self._saver.restore(self._session, path)