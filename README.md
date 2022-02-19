# ray-bugfix
This is a workaround to issues with RLLib, such as https://github.com/ray-project/ray/issues/15089
Though it can be used for any gym environment, with very small changes

# Information for running scripts
The NN models can be changed or fixed in the model.py file

The update algorithm can be added similarly to the DQN.py and PPO.py files (please stay consistent when doing so)

To change the task, change the worker.py and episodeTest.py - performNActions function

The hyperparameters, are in the runner.py file, with the scripts being run from there

A more cluttered version of this can be found in the rayParameters.py file
