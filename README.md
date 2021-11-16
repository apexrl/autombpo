# On Effective Scheduling of Model-based Reinforcement Learning

Code to reproduce the experiments in On Effective Scheduling of Model-based Reinforcement Learning.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

Mujoco license is required to run the experiments on the Mujoco environments.
## Training

To train the hyper-controller of the paper, run this command:

```train
python train.py --env=<env_name>
```

The env_name can be selected from [hopper,ant,humanoid,hopperbullet,walker2dbullet,halfcheetahbullet]. For example: python train.py --env=hopper

The trained hyper-controller will be saved in saved-models/. The computing infrastructure used in our experiments and the around computation time to train the hyper-controller is provided in Appendix G.

## Evaluation

**After training**, to evaluate the trained hyper-controller, run:

```eval
python eval.py --config=config.<env_name> --model_path=saved-models
```
The env_name can be selected from [hopper,ant,humanoid,hopperbullet,walker2dbullet,halfcheetahbullet]. For example: python eval.py --config=config.hopper --model_path=saved-models

**Notice this command can only be run after finishing training the hyper-controller on the corresponding environments.**


## Pre-trained Models
We provided our pre-trained hyper-controller in pre-trained-models/ to better reproduce the experiments. To evaluate the pre-trained models, run:
```eval
python eval.py --config=config.<env_name> --model_path=pre-trained-models
```
The env_name can be selected from [hopper,ant,humanoid,hopperbullet,walker2dbullet,halfcheetahbullet]. For example: python eval.py --config=config.hopper --model_path=pre-trained-models