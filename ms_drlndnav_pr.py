# * ---------------- *
#
#   ** Deep Reinforcement Learning Nano Degree **
#   project: Navigation
#   author:  Matthias Schinacher
#
#   the script implements Q-learning with transitions- replay
#   optionally using priority replay
#
#   the model used for the Q- function approximation is a simple
#   neural network with 3 hidden layers and 2 'relu' activations in between;
#   note: there is no activation/ relu after the third linear function!
#
#   => thus the model has 2 adjustable size- parameters, as the number of
#      incoming values is determined by the sate- size (37) and the number
#      of outgoing values ("activations") is the number of actions (4)
#   => interpretation of the 4 resulting values is simply the Q- values
#      for the 4 actions (of the input-state given)
# * ---------------- *

# * ---------------- *
#    importing the packages we need
# * ---------------- *
import os.path
import sys
import copy
import re
import configparser
import pickle
from unityagents import UnityEnvironment
import numpy as np
import torch
import torch.nn.functional as fct

# * ---------------- *
#   command line arguments:
#    we expect exactly 2, the actual script name and the command-file-name
# * ---------------- *
if len(sys.argv) != 2:
    print('usage:')
    print('   python {} command-file-name'.format(sys.argv[0]))
    quit()

if not os.path.isfile(sys.argv[1]):
    print('usage:')
    print('   python {} command-file-name'.format(sys.argv[0]))
    print('[error] "{}" file not found or not a file!'.format(sys.argv[1]))
    quit()

# * ---------------- *
#   constants:
#    this code is only for the Banana- scenario, no generalization (yet)
# * ---------------- *
STATE_SIZE = int(37)
ACTION_SIZE = int(4)
fzero = float(0)
fone = float(1)

# * ---------------- *
#   the command-file uses the ConfigParser module, thus must be structured that way
#    => loading the config and setting the respective script values
# * ---------------- *
booleanpattern = re.compile('^\\s*(true|yes|1|on)\\s*$', re.IGNORECASE)

config = configparser.ConfigParser()
config.read(sys.argv[1])

# start the logfile
rlfn = 'run.log' # run-log-file-name
if 'global' in config and 'runlog' in config['global']:
    rlfn = config['global']['runlog']
print('!! using logfile "{}"\n'.format(rlfn))
rl = open(rlfn,'w')
rl.write('# ## configuration from "{}"\n'.format(sys.argv[1]))

if 'rand' in config and 'seed' in config['rand']:
    seed = int(config['rand']['seed'])
    rl.write('# [debug] using random seed: {}\n'.format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)

TRAIN = True   # default to training mode
if 'mode' in config and 'train' in config['mode']:
    train = config['mode']['train']
    TRAIN = True if booleanpattern.match(train) else False
    rl.write('# [debug] using mode.train: {} from "{}"\n'.format(TRAIN,train))
SHOW  = not TRAIN  # default for "show"- mode
if 'mode' in config and 'show' in config['mode']:
    show = config['mode']['show']
    SHOW = True if booleanpattern.match(show) else False
    rl.write('# [debug] using mode.show: {} from "{}"\n'.format(SHOW,show))

# * ---------------- *
#   hyper- parameters
# * ---------------- *
# defaults
EPISODES = int(1000)         # number of episodes (including warm-up)
WARMUP_EPISODES = int(10)    # number of warm-up episodes (training only)
EPSILON_EPISODES = int(100)  # number of episodes, over which we decrease epsilon (training only)
EPSILON_START = float(1)     # start-value for epsilon (training and show- mode!)
EPSILON_END = float(0.01)    # final value for epsilon (training only)
REPLAY_BUFFERSIZE = int(50000) # replay buffer/memory- size (training only)
REPLAY_BATCHSIZE = int(300)  # batch size for replay (training only)
REPLAY_STEPS = int(1)        # replay transisitions- batch each x steps (training only)
PRIO_REPLAY = False          # priority replay flag (training only)
Q_RESET_STEPS = int(50)      # steps, after which to reset Q_ to Q  (training only)
GAMMA = float(0.99)          # gamma- parameter (training only)
LEARNING_RATE = float(0.001) # (training only)

# overwrite defaults
if 'hyperparameters' in config:
    hp = config['hyperparameters']
    EPISODES          = int(hp['episodes'])          if 'episodes'          in hp else EPISODES
    WARMUP_EPISODES   = int(hp['warmup_episodes'])   if 'warmup_episodes'   in hp else WARMUP_EPISODES
    EPSILON_EPISODES  = int(hp['epsilon_episodes'])  if 'epsilon_episodes'  in hp else EPSILON_EPISODES
    EPSILON_START     = float(hp['epsilon_start'])   if 'epsilon_start'     in hp else EPSILON_START
    EPSILON_END       = float(hp['epsilon_end'])     if 'epsilon_end'       in hp else EPSILON_END
    REPLAY_BUFFERSIZE = int(hp['replay_buffersize']) if 'replay_buffersize' in hp else REPLAY_BUFFERSIZE
    REPLAY_BATCHSIZE  = int(hp['replay_batchsize'])  if 'replay_batchsize'  in hp else REPLAY_BATCHSIZE
    REPLAY_STEPS      = int(hp['replay_steps'])      if 'replay_steps'      in hp else REPLAY_STEPS
    if 'prio_replay' in hp:
        PRIO_REPLAY   = True if booleanpattern.match(hp['prio_replay']) else False
    Q_RESET_STEPS     = int(hp['q_reset_steps'])     if 'q_reset_steps'     in hp else Q_RESET_STEPS
    GAMMA             = float(hp['gamma'])           if 'gamma'             in hp else GAMMA
    LEARNING_RATE     = float(hp['learning_rate'])   if 'learning_rate'     in hp else LEARNING_RATE

# model- defaults (only if model is not loaded from file)
MODEL_H1 = int(10)     # hidden layer size 1
MODEL_H2 = int(10)     # hidden layer size 2

# filenames for loading the model and buffer/memory of transistions
load_file = 'Q.model' if not TRAIN else None # only default when not training
load_transitions_file = None
# filenames for saving the model (and the "best model") and buffer/memory of transistions
save_file = 'Q.out.model' if TRAIN else None # only default when training
save_best_file = 'Q.out.best.model' if TRAIN else None # only default when training
save_transitions_file = None

# overwrite defaults
if 'model' in config:
    m = config['model']
    MODEL_H1 = int(m['h1'])  if 'h1' in m else MODEL_H1
    MODEL_H2 = int(m['h2'])  if 'h2' in m else MODEL_H2
    load_file = m['load_file']                         if 'load_file'             in m else load_file
    load_transitions_file = m['load_transitions_file'] if 'load_transitions_file' in m else load_transitions_file
    save_file = m['save_file']                         if 'save_file'             in m else save_file
    save_best_file = m['save_best_file']               if 'save_best_file'        in m else save_best_file
    save_transitions_file = m['save_transitions_file'] if 'save_transitions_file' in m else save_transitions_file

# consistency check, reset values not consistent
if EPSILON_EPISODES <= 0:
    EPSILON_EPISODES = 0
    EPSILON_START = EPSILON_END
    DELTA_EPSILON = 0.0
# computed
else:
    DELTA_EPSILON = (EPSILON_START - EPSILON_END)/float(EPSILON_EPISODES +1)

# * ---------------- *
#   writing the used config to the logfile
# * ---------------- *
rl.write('# TRAIN (mode):      {}\n'.format(TRAIN))
rl.write('# SHOW (mode):       {}\n\n'.format(SHOW))
rl.write('# EPSILON_EPISODES:  {}\n'.format(EPSILON_EPISODES))
rl.write('# EPISODES:          {}\n'.format(EPISODES))
rl.write('# WARMUP_EPISODES:   {}\n'.format(WARMUP_EPISODES))
rl.write('# EPSILON_START:     {}\n'.format(EPSILON_START))
rl.write('# EPSILON_END:       {}\n'.format(EPSILON_END))
rl.write('# DELTA_EPSILON:     {}\n'.format(DELTA_EPSILON))
rl.write('# REPLAY_BUFFERSIZE: {}\n'.format(REPLAY_BUFFERSIZE))
rl.write('# REPLAY_BATCHSIZE:  {}\n'.format(REPLAY_BATCHSIZE))
rl.write('# REPLAY_STEPS:      {}\n'.format(REPLAY_STEPS))
rl.write('# PRIO_REPLAY:       {}\n'.format(PRIO_REPLAY))
rl.write('# Q_RESET_STEPS:     {}\n'.format(Q_RESET_STEPS))
rl.write('# GAMMA:             {}\n'.format(GAMMA))
rl.write('# LEARNING_RATE:     {}\n'.format(LEARNING_RATE))
rl.write('#   -- model\n')
rl.write('# H1: {}\n'.format(MODEL_H1))
rl.write('# H2: {}\n'.format(MODEL_H2))
rl.write('# load_file: {}\n'.format(load_file))
rl.write('# load_transitions_file: {}\n'.format(load_transitions_file))
rl.write('# save_file: {}\n'.format(save_file))
rl.write('# save_best_file: {}\n'.format(save_best_file))
rl.write('# save_transitions_file: {}\n'.format(save_transitions_file))
rl.flush()

# * ---------------- *
#   torch:
#    local computer was a laptop with no CUDA available
#    => feel free to change this, if you have a machine (with GPU)
# * ---------------- *
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# * ---------------- *
#   buildung and initializing the torch- models for Q and Q_
# * ---------------- *
if load_file and os.path.isfile(load_file):
    modelQ = torch.load(load_file)
    modelQ_ = torch.load(load_file) # instead of zeros, we initialize with the same model

    rl.write('# [info] loaded model(s) from "{}"\n'.format(load_file))
    rl.write('# [info]   => H1, H2 parameters not used (nn-sizes from file model are implicitly used)\n')
    rl.flush()
elif not TRAIN:
    rl.write('# [error] not training, but model not loaded from file ("{}").\n'.format(load_file))
    rl.close()
    quit()
else:
    L1 = torch.nn.Linear(STATE_SIZE, MODEL_H1)
    torch.nn.init.uniform_(L1.weight) # uniform [0,1) for the actual Q function
    torch.nn.init.uniform_(L1.bias)
    L2 = torch.nn.Linear(MODEL_H1, MODEL_H2)
    torch.nn.init.uniform_(L2.weight)
    torch.nn.init.uniform_(L2.bias)
    L3 = torch.nn.Linear(MODEL_H2, ACTION_SIZE)
    torch.nn.init.uniform_(L3.weight)
    torch.nn.init.uniform_(L3.bias)

    modelQ = torch.nn.Sequential(
        L1
        ,torch.nn.ReLU()
        ,L2
        ,torch.nn.ReLU()
        ,L3
    #    ,torch.nn.LogSoftmax()
    )

    L1_ = torch.nn.Linear(STATE_SIZE, MODEL_H1)
    torch.nn.init.constant_(L1_.weight,fzero) # zeros, for the fixed Q
    torch.nn.init.constant_(L1_.bias,fzero)
    L2_ = torch.nn.Linear(MODEL_H1, MODEL_H2)
    torch.nn.init.constant_(L2_.weight,fzero)
    torch.nn.init.constant_(L2_.bias,fzero)
    L3_ = torch.nn.Linear(MODEL_H2, ACTION_SIZE)
    torch.nn.init.constant_(L3_.weight,fzero)
    torch.nn.init.constant_(L3_.bias,fzero)

    modelQ_ = torch.nn.Sequential(
        L1_
        ,torch.nn.ReLU()
        ,L2_
        ,torch.nn.ReLU()
        ,L3_
    #    ,torch.nn.LogSoftmax()
    )

    L1.weight.requires_grad_(requires_grad=True)
    L2.weight.requires_grad_(requires_grad=True)
    L3.weight.requires_grad_(requires_grad=True)
    L1.bias.requires_grad_(requires_grad=True)
    L2.bias.requires_grad_(requires_grad=True)
    L3.bias.requires_grad_(requires_grad=True)

    L1_.weight.requires_grad_(requires_grad=False)
    L2_.weight.requires_grad_(requires_grad=False)
    L3_.weight.requires_grad_(requires_grad=False)
    L1_.bias.requires_grad_(requires_grad=False)
    L2_.bias.requires_grad_(requires_grad=False)
    L3_.bias.requires_grad_(requires_grad=False)

if TRAIN:
    optimizer = torch.optim.Adam(modelQ.parameters(),lr=LEARNING_RATE)

# * ---------------- *
#   loading the Banana environment, loading the default brain (external)
# * ---------------- *
env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# * ---------------- *
#   the actual algorithm
# * ---------------- *

if TRAIN:
    # a very simple replay memory, a list (of tuples)
    #   - is assumed to never shrink
    #   - only inserts at given index, if next index would be > size, start at 0
    #         => list entries 0..size-1 are occupied
    replay_memory = []   # actual replay memory
    p_array       = []   # priority array
    if PRIO_REPLAY:
        index_array   = [i for i in range(REPLAY_BUFFERSIZE)]
        min_p         = abs(0.00000001) # value added to |delta| for numeric stability
    rm_size = 0          # number of entries in replay memory
    rm_next = 0          # next index to use for insert

    # we always laod/save transitions together with the priority- p_array
    #  ... even if we do not or had not used priority replay
    if load_transitions_file and os.path.isfile(load_transitions_file):
        with open(load_transitions_file, 'rb') as f:
            ( tmpm, tmpp ) = pickle.load(f)

            replay_memory = tmpm if REPLAY_BUFFERSIZE >= len(tmpm) else tmpm[0:REPLAY_BUFFERSIZE]
            rm_size = len(replay_memory)
            rm_next = rm_size if rm_size < REPLAY_BUFFERSIZE else 0

            if PRIO_REPLAY:
                # we need to make certain, that p_array has the same length
                #  as the replay-memory (we pad with the average if necc.)
                p_array = tmpp if tmpp else []
                if len(p_array) > rm_size:
                    p_array = p_array[0:rm_size]
                elif len(p_array) == 0:
                    if rm_size > 0:
                        p_array = [float(1.0) for i in range(rm_size)]
                elif len(p_array) < rm_size:
                    ptmpv = float(sum(p_array))/float(len(p_array))
                    tmplen = len(p_array)
                    for i in range(tmplen,rm_size):
                        p_array.append(ptmpv)
            else:
                p_array = [] # discard if we don't use prio-replay

# score buffer, for the last 100 scores
score_buffer = []
max_score = float('-inf')
best_model = None

epsilon = EPSILON_START
q_steps = 0
r_steps = 0

rl.write('#\n# Episode Score average(last-100-Scores) Steps Epsilon\n')

for episode in range(1,EPISODES+1):
    train_mode = not SHOW
    env_info = env.reset(train_mode=train_mode)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the start state
    score = 0                                          # initialize the episode- score
    step  = 0                                          # step within episodes
    if TRAIN and episode > WARMUP_EPISODES and (epsilon - DELTA_EPSILON + 1e-10) >= EPSILON_END:
        epsilon -= DELTA_EPSILON
    while True:
        step += 1
        rs = np.random.random_sample()
        if TRAIN and episode <= WARMUP_EPISODES:
            action = np.random.randint(ACTION_SIZE)
        elif (epsilon >= fone or rs < epsilon):
            action = np.random.randint(ACTION_SIZE)
        else:
            pred_a = modelQ(torch.tensor(state))
            action = int(torch.argmax(pred_a))

        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished

        #print("Episode: {}; Step: {}; Reward: {}".format(episode,step,reward))

        if TRAIN:
            if PRIO_REPLAY:
                p = abs(float( reward + GAMMA * max(modelQ(torch.tensor(next_state))) - modelQ(torch.tensor(state))[action] )) + min_p
            # store transition in replay memory
            transition = (state,action,reward,next_state,done)
            if rm_size < REPLAY_BUFFERSIZE:
                replay_memory.append(transition)
                if PRIO_REPLAY:
                    p_array.append(p)
                rm_size += 1
            else:
                replay_memory[rm_next] = transition
                if PRIO_REPLAY:
                    p_array[rm_next] = p
            rm_next += 1
            if rm_next >= REPLAY_BUFFERSIZE:
                rm_next = 0

            if rm_size >= REPLAY_BATCHSIZE and episode > WARMUP_EPISODES:
                r_steps += 1
                if r_steps >= REPLAY_STEPS:
                    r_steps = 0

                    # learn with random sampled transitions from the past
                    if PRIO_REPLAY:
                        p_sum = float(sum(p_array))
                        tmp = float(1)/p_sum
                        P = np.array(p_array) * tmp
                        #print('[DEBUG] p_array: ',p_array)
                        #print('[DEBUG] P: ',P)
                        if rm_size < REPLAY_BUFFERSIZE:
                            batch_idx = np.random.choice(index_array[0:rm_size],size=REPLAY_BATCHSIZE,p=P)
                        else:
                            batch_idx = np.random.choice(index_array,size=REPLAY_BATCHSIZE,p=P)
                    else:
                        batch_idx = np.random.randint(rm_size, size=REPLAY_BATCHSIZE)
                    #print('[DEBUG] batch_idx: ',batch_idx)

                    listt  = [replay_memory[idx] for idx in batch_idx] # transitions
                    lists  = torch.tensor([s for s,a,r,ns,d in listt],dtype=torch.float64) # states
                    lista  = torch.tensor([a for s,a,r,ns,d in listt],dtype=torch.int64) # actions
                    listr  = torch.tensor([[r] for s,a,r,ns,d in listt],dtype=torch.float64) # rewards
                    listns = torch.tensor([ns for s,a,r,ns,d in listt],dtype=torch.float64) # next states
                    listd  = torch.tensor([[d] for s,a,r,ns,d in listt],dtype=torch.int64) # "done"- flags
                    tmpzeroes = torch.tensor([0 for i in range(0,len(batch_idx))],dtype=torch.int64)
                    tmpa   = torch.stack([lista] + [tmpzeroes for i in range(1,ACTION_SIZE)])
                    tmpa   = torch.t(tmpa)

                    pred_v = torch.index_select(modelQ(lists).gather(1, tmpa), 1, torch.tensor([0]))

                    pred_v_, _ = torch.max(modelQ_(listns),dim=1,keepdim=True)
                    pred_v_ =  listr  + pred_v_ * GAMMA * torch.tensor((1 - listd),dtype=torch.float64)

                    #quit()

                    loss = fct.mse_loss(pred_v,pred_v_)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step

        if TRAIN:
            q_steps += 1
            if q_steps >= Q_RESET_STEPS:
                modelQ_.load_state_dict(modelQ.state_dict())
                q_steps = 0

        if done:                                       # exit loop if episode finished
            break

    if max_score < score:
        max_score = score
        best_model = copy.deepcopy(modelQ)

    score_buffer.append(score)
    while len(score_buffer) > 100:
        score_buffer.pop(0)
    l100_score = float(sum(score_buffer))/float(len(score_buffer)) if len(score_buffer) >= 100 else float(0)

    rl.write('{} {} {} {} {}\n'.format(episode,score,l100_score,step,epsilon))
    rl.flush()
    print("Episode: {}; Score: {} ({}); #Steps: {}; Epsilon: {}".format(episode,score,l100_score,step,epsilon))

env.close()

if TRAIN:
    if save_file:
        rl.write('# .. writing final model to "{}"\n'.format(save_file))
        torch.save(modelQ,save_file)
    if save_best_file:
        rl.write('# .. writing best model to "{}"\n'.format(save_best_file))
        torch.save(best_model,save_best_file)

    if save_transitions_file:
        rl.write('# .. saving transisitions to "{}"\n'.format(save_transitions_file))
        with open(save_transitions_file, 'wb') as f:
            pickledata = ( replay_memory,p_array )
            pickle.dump(pickledata, f, pickle.HIGHEST_PROTOCOL)

rl.close()
