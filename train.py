import argparse
import numpy as np
import tensorflow as tf
from controller import Controller
import os
import threading
import time
import random

def delete_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            delete_file(c_path)
            os.rmdir(c_path)
        else:
            os.remove(c_path)

def run_an_exp(env = None,exp_id = 0,epsilon=0):
    os.system('python main.py --config=config.%s --exp_id=%d --epsilon=%f' %(env,exp_id,epsilon))


#Set kwargs for hyper-controller training
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='hopper')

args = parser.parse_args()
env = args.env


STAGE_NUM = {
    'hopper': int(10),
    'ant': int(10),
    'humanoid': int(20),
    'hopperbullet': int(20),
    'walker2dbullet': int(20),
    'halfcheetahbullet':int(10),
}[env]

EPSILON_SCHEDULE = {
    'hopper': [0,5,1,0],
    'ant': [0,5,1,0],
    'humanoid': [0,5,1,0],
    'hopperbullet': [0,5,0,0],
    'walker2dbullet': [0,5,0,0],
    'halfcheetahbullet':[0,5,0,0],
}[env]


DOMAIN = {
    'hopper': 'Hopper',
    'ant': 'Ant',
    'humanoid': 'Humanoid',
    'hopperbullet': 'HopperBulletEnv',
    'walker2dbullet': 'Walker2DBulletEnv',
    'halfcheetahbullet':'HalfCheetahBulletEnv',
}[env]

HYPERPARAMETERS_STES = {
    'Hopper': [['model', 'ratio']],
    'Ant': [['model', 'ratio']],
    'Humanoid': [['model', 'ratio']],
    'HopperBulletEnv': [['model', 'policy','ratio']],
    'Walker2DBulletEnv': [['model', 'policy','ratio']],
    'HalfCheetahBulletEnv': [['model', 'policy','ratio']],
}[DOMAIN]

CONTROLLERS_INIT = {
    'Hopper': [True, False, False],
    'Ant': [False, False, False],
    'Humanoid': [True, False, False],
    'HopperBulletEnv': [False],
    'Walker2DBulletEnv': [False],
    'HalfCheetahBulletEnv': [False],
}[DOMAIN]



EPISODE_PER_STAGE = 10
UPDATE_PER_EPISODE = 30
BATCH_SIZE = 64

#create folder to store the trained model and temp data
model_path = 'saved-models/' + DOMAIN + '/controller0'
buffer_path = 'buffer'
log_path = 'log/' + DOMAIN
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(buffer_path):
    os.makedirs(buffer_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)
delete_file(model_path)
delete_file(buffer_path)

current_model_path =  model_path + '/current'
best_model_path = model_path + '/best'


hyperparameters = HYPERPARAMETERS_STES[0]
controllers_init = CONTROLLERS_INIT[0]
state_dim = 4 + len(hyperparameters)
controller_graph = tf.Graph()
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
controller_session = tf.InteractiveSession(config=config,graph=controller_graph)
action_dim = len(hyperparameters)
action_space = []
for hyperparameeter in hyperparameters:
    if (hyperparameeter == 'model'):
        action_space.append(2)
    else:
        action_space.append(3)
controller = Controller(state_dim=state_dim, action_dim=action_dim, action_space=action_space,
                        hyperparameters=hyperparameters,init = controllers_init,
                        graph=controller_graph,session=controller_session)
controller.save(path=current_model_path)


best_mbrl_return = -1e9
threads = []

for stage in range(STAGE_NUM):
    min_stage, max_stage, max_epsilon, min_epsilon = EPSILON_SCHEDULE
    dx = (stage - min_stage) / (max_stage - min_stage)
    y = dx * (min_epsilon - max_epsilon) + max_epsilon
    epsilon = max(y,0)
    for i in range(EPISODE_PER_STAGE):
        job = lambda: run_an_exp(env=env,exp_id=stage*EPISODE_PER_STAGE+i,epsilon=epsilon)
        t = threading.Thread(target=job)
        t.start()
        threads.append(t)
        time.sleep(random.randint(30,60))

    for t in threads:
        t.join()

    mbrl_returns = []
    for i in range(EPISODE_PER_STAGE):
        # read data from the buffer
        if(os.path.exists('./buffer/states_%d.npy' % (stage*EPISODE_PER_STAGE+i))):
            states = np.load('./buffer/states_%d.npy' % (stage*EPISODE_PER_STAGE+i))
            actions = np.load('./buffer/actions_%d.npy' % (stage*EPISODE_PER_STAGE+i))
            advs = np.load('./buffer/advs_%d.npy' % (stage*EPISODE_PER_STAGE+i))
            logps = np.load('./buffer/logps_%d.npy' % (stage*EPISODE_PER_STAGE+i))
            controller.store(states, actions, advs, logps)
            mbrl_return = np.load('./buffer/mbrl_returns_%d.npy' % (stage*EPISODE_PER_STAGE+i))
            mbrl_returns.append(np.mean(mbrl_return))

    if(np.mean(mbrl_returns) > best_mbrl_return):
        controller.save(path=best_model_path)
        best_mbrl_return = np.mean(mbrl_returns)

    controller.update(gradient_steps=UPDATE_PER_EPISODE*EPISODE_PER_STAGE,batch_size=BATCH_SIZE)
    controller.save(path=current_model_path)
    controller.clear()

delete_file(log_path)