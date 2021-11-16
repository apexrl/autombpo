import argparse
import importlib
import runner
import os
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--exp_id', type=int, default=0)
parser.add_argument('--epsilon', type=float, default=0.0)
parser.add_argument('--model_path', type=str, default='saved-models')

# parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

exp_id = args.exp_id
epsilon = args.epsilon
model_path = args.model_path


module = importlib.import_module(args.config)
params = getattr(module, 'params')
universe, domain, task = params['universe'], params['domain'], params['task']

NUM_EPOCHS_PER_DOMAIN = {
    'Hopper': int(50),
    'Ant': int(50),
    'Humanoid': int(100),
    'HopperBulletEnv': int(30),
    'Walker2DBulletEnv': int(50),
    'HalfCheetahBulletEnv':int(20),
}[domain]

MAX_STEPS = {
    'Hopper': int(1000),
    'Ant': int(1000),
    'Humanoid': int(1000),
    'HopperBulletEnv': int(1000),
    'Walker2DBulletEnv': int(1000),
    'HalfCheetahBulletEnv':int(1000),
}[domain]

EXPLORATION_STEPS = {
    'Hopper': int(5000),
    'Ant': int(5000),
    'Humanoid': int(5000),
    'HopperBulletEnv': int(5000),
    'Walker2DBulletEnv': int(5000),
    'HalfCheetahBulletEnv':int(5000),
}[domain]

HYPERPARAMETERS_STES = {
    'Hopper': [['model', 'ratio'], ['policy'],['rollout']],
    'Ant': [['model', 'ratio'], ['policy'], ['rollout']],
    'Humanoid': [['model', 'ratio'], ['policy'], ['rollout']],
    'HopperBulletEnv': [['model', 'policy','ratio']],
    'Walker2DBulletEnv': [['model', 'policy','ratio']],
    'HalfCheetahBulletEnv': [['model', 'policy','ratio']],
}[domain]

CONTROLLERS_INIT = {
    'Hopper': [True, False, False],
    'Ant': [False, False, False],
    'Humanoid': [True, False, False],
    'HopperBulletEnv': [False],
    'Walker2DBulletEnv': [False],
    'HalfCheetahBulletEnv': [False],
}[domain]

if(model_path == 'saved-models'):
    HYPERPARAMETERS_STES = HYPERPARAMETERS_STES[:1]
    CONTROLLERS_INIT = CONTROLLERS_INIT[:1]

params['kwargs']['n_epochs'] = NUM_EPOCHS_PER_DOMAIN
params['kwargs']['n_initial_exploration_steps'] = EXPLORATION_STEPS
params['kwargs']['reparameterize'] = True
params['kwargs']['lr'] = 3e-4
params['kwargs']['target_update_interval'] = 1
params['kwargs']['tau'] = 5e-3
params['kwargs']['store_extra_policy_info'] = False
params['kwargs']['action_prior'] = 'uniform'

variant_spec = {
        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': {},
            },
            'evaluation': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': {},
            },
        },
        'policy_params': {
            'type': 'GaussianPolicy',
            'kwargs': {
                'hidden_layer_sizes': (256, 256),
                'squash': True,
            }
        },
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (256, 256),
            }
        },
        'controller_params': {
            'hyperparameters_set':HYPERPARAMETERS_STES,
            'controllers_init':CONTROLLERS_INIT,
        },
        'algorithm_params': params,
        'replay_pool_params': {
            'type': 'ExtraPolicyInfoReplayPool',
            'kwargs': {
                'max_size': int(1e6),
            }
        },
        'sampler_params': {
            'type': 'ExtraPolicyInfoSampler',
            'kwargs': {
                'max_path_length': MAX_STEPS,
                'min_pool_size': MAX_STEPS,
                'batch_size': 256,
            }
        },
        'run_params': {
            # 'seed': args.seed,
            'mode':'train',
            'model_path':model_path,
            'exp_id':exp_id,
            'epsilon':epsilon,
            'checkpoint_at_end': True,
            'checkpoint_frequency': NUM_EPOCHS_PER_DOMAIN // 10,
            'checkpoint_replay_pool': False,
        },
    }

exp_runner = runner.ExperimentRunner(variant_spec)
diagnostics = exp_runner.train()
