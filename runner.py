#Ray parameterserver and worker
from ParameterServer import ParameterServer
from worker import DataWorker
import ray
from episodeTest import performNActions

#Update algorithm (PPO, DQN, DDPG, ACKTR ++)
def updateAlgo(Algo):
    if Algo == "DQN":
        action_space = "<class 'gym.spaces.discrete.Discrete'>"  # Discrete - DQN
        from DQN import DQN #Algorithm used for the backpropagation
        algo = DQN

    elif Algo == "PPO":
        action_space = "<class 'gym.spaces.box.Box'>" #Continuous - PPO
        from PPO import PPO  # Algorithm used for the backpropagation
        algo = PPO

    else:
        raise Exception

    return algo, action_space

DQNPPO = "PPO"
algo, action_space = updateAlgo(DQNPPO)

#Create the environment
from Environment import createEnv
environmentName = "CarRacing-v0"
env, obs = createEnv(environmentName)#env, obs = createEnv("CartPole-v0")

#Get statespace
def getStateSpace(obs, output):
    try:
        x = len(obs)
        assert type(x) == type(0)
        output.append(x)
        getStateSpace(obs[0], output)

    except:
        print("State space is: {} ".format(output))
    return output

#number of actions, also checks if it is discrete or continuous
def getActionSpace(env, discreteContinuous):
    action_space = env.action_space
    assert str(type(action_space)) == discreteContinuous #Check that the environment is the same as your algorithm
    #                                                                       (compatibility)

    if discreteContinuous == "<class 'gym.spaces.discrete.Discrete'>":
        output = action_space.n
    else:
        output = action_space.shape[0]
    return output

#state_dim, action_dim, n_latent_var, lr, betas, gamma, updateAlgo, model, device, env
#HyperParameters
state_dim = getStateSpace(obs, []) #Can be image or state
action_dim = getActionSpace(env, action_space) #Gives the action dimmension
hidden_layer = 64 #Only used for FCN
lr = 0.002
betas = (0.9, 0.999)#Adam params
gamma = 0.9         #given a decay rate
K_epochs = 4        #PPO number of times to run through the data
eps_clip = 0.2      #PPO clip rate, 0.2-0.3 gives the best results
device = "cpu"

#Ray hyperparameter
num_workers = 4

#Number of episodes
episode_num = 100


#Model
from model import *
# model = ConvDQN(state_dim[-1], action_dim, hidden_layer)
model = ConvActorCritic
# model = PPOFCN(stateDim, action_dim, hidden_layer)
# model = DQNFCN(stateDim, action_dim, hidden_layer)


ray.init(ignore_reinit_error=True)
ps = ParameterServer.remote(state_dim, action_dim, hidden_layer, lr, betas, gamma)
workers = [DataWorker.remote(state_dim, action_dim, hidden_layer, lr, betas, gamma, algo, model, device, environmentName) for i in range(num_workers)]

current_weights = ps.get_weights.remote()
def runner():
    print("Running synchronous parameter server training.")
    current_weights = ps.get_weights.remote()
    for i in range(episode_num % num_workers):
        #Give the agents their task
        gradients = [
            worker.compute_gradients.remote(current_weights) for worker in workers
        ]

        # Calculate update after all gradients are available.
        current_weights = ps.apply_gradients.remote(*gradients)
        print("Done running workers")

        # Evaluate the current model.
        model.policy_net.set_weights(ray.get(current_weights))
        reward = performNActions(model, 1000, env)
        print("Iter {}: \t reward is {:.4f}".format(i, reward))

runner()