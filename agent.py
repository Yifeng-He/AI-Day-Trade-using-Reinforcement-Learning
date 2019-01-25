
import pandas as pd
import numpy as np
import tensorflow as tf

import gym
import time
import random
import threading

from keras import layers
from keras.models import Model
from keras import utils as np_utils
from keras import optimizers
from keras.models import Sequential
from keras.layers import Merge, Conv1D, Dense, Dropout, Flatten, concatenate, Input

#from keras.models import *
#from keras.layers import *
from keras import backend as K
from market_env_SPY_v5 import MarketEnv


#-- constants
#ENV = 'CartPole-v0'

RUN_TIME = 30
THREADS = 1
OPTIMIZERS = 1
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 16  # 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP = .15
EPS_STEPS = 75000

MIN_BATCH = 10000  # 32 # this is actual the sample size
BATCH_SIZE = 200
NUM_EPOCH = 20 # added by Yifeng, for mini-batch learning
NUM_GENERATIONS = 50000
SAVED_EPISODES = 10000


LEARNING_RATE = 5e-3

LOSS_V = .5			# v loss coefficient
LOSS_ENTROPY = .01 	# entropy coefficient

NUM_PRICE_STATE = 60
NUM_VOL_STATE = 60



#---------


class Brain:
    train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()
    # track the results
    GLOABL_LOSS_VECTOR =[]
    GLOBAL_IND_EPOCH = 0
    GLOBAL_EPISODE_REWARDS =[]

    def __init__(self):
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)
        # the NN model: state-->16 neurons-->output1(num_states) + output2(state-value)
#        self.model = self._build_model()
        self.model = self._build_conv1D_model()
        # computational graph based on the NN model
#        self.graph = self._build_graph(self.model)
        self.graph = self._build_conv1D_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()  # avoid modifications

    
    def _build_conv1D_model(self):
        #### the model
        x_p = Input(batch_shape=(None, NUM_PRICE_STATE, 1)) #shape=(1, 60, 1)
        x_v = Input(batch_shape=(None, NUM_VOL_STATE, 1))  #shape=(1, 60, 1)

        x1 = Conv1D(filters=500, kernel_size=2, activation='relu')(x_p)
        x1_out1 = Conv1D(filters=500, kernel_size=4, activation='relu')(x1) 
        x1_out = Conv1D(filters=500, kernel_size=6, activation='relu')(x1_out1)
        x1_f = Flatten()(x1_out)
        
        x2 = Conv1D(filters=500, kernel_size=2, activation='relu')(x_v)
        x2_out1 = Conv1D(filters=500, kernel_size=4, activation='relu')(x2) 
        x2_out = Conv1D(filters=500, kernel_size=6, activation='relu')(x2_out1)
        x2_f = Flatten()(x2_out)
         
        x_12 = concatenate([x1_f, x2_f])
        x_a = Dense(600, activation='relu')(x_12)
        out_p = Dense(NUM_ACTIONS, activation='softmax')(x_a)
        out_v = Dense(1, activation='linear')(x_a)
        
        model = Model(inputs=[x_p, x_v], outputs=[out_p, out_v])
        model._make_predict_function()  # have to initialize before threading
        return model 

    def _build_conv1D_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATE, 1))
        a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        # not immediate, but discounted n step reward
        r_t = tf.placeholder(tf.float32, shape=(None, 1))
        
        x_p = tf.slice(s_t, [0, 0, 0], [-1, NUM_VOL_STATE, -1])
        #x_p = tf.expand_dims(x, -1) #[batch_size, 60, 1]
        x_v = tf.slice(s_t, [0, NUM_VOL_STATE, 0], [-1, NUM_VOL_STATE, -1]) 
        
        p, v = model([x_p, x_v])
        # log-probability for the action taken
        log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = r_t - v  # q-value - state-value

        # maximize policy
        loss_policy = - log_prob * tf.stop_gradient(advantage)		
        # minimize value error
        loss_value = LOSS_V * tf.square(advantage)
        # maximize entropy (regularization)
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

#        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
#        optimizer = tf.train.AdamOptimizer(0.01)
        optimizer = tf.train.RMSPropOptimizer(0.01, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, minimize, loss_total


    def optimize(self):  # train the model once using a batch of samples
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)  # yield
            return
        # get >= a batch of samples from queue
        with self.lock_queue:
            # more thread could have passed without lock
            if len(self.train_queue[0]) < MIN_BATCH:
                return 									# we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [[], [], [], [], []]
            # now we have the batch of the data
        # convert s from a list of lists into a 2D numpy array
        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5 * MIN_BATCH:
            print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state

        # s_t, a_t, r_t are placeholders, minmize is train_step
        s_t, a_t, r_t, minimize, tot_loss = self.graph
        # update the NN weights once
        # do mini-batch training for num_epoch ***********************
#        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})
        len_data = len(r)
        for i in range(NUM_EPOCH):
            loss_epoch = 0
            # shuffle the training data
            rand_index = np.random.choice(len_data, len_data, replace=False)
            rand_s = s[rand_index]
            rand_a = a[rand_index]
            rand_r = r[rand_index]
            
            #get a mini-batch
            batch_ind = 0
            for k in range(0, len_data, BATCH_SIZE):
                batch_ind += 1
                end_point = k+BATCH_SIZE if (k+BATCH_SIZE) < len_data else len_data
                batch_s = rand_s[k:end_point]
                batch_s = np.expand_dims(batch_s, -1)
                batch_a = rand_a[k:end_point]
                batch_r = rand_r[k:end_point]
#                print('batch_length=%d' % len(batch_s))
#                print(batch_s.shape)
#                print(batch_a.shape)
                # train with a mini-batch data
                _, temp_loss = self.session.run([minimize, tot_loss], feed_dict={s_t: batch_s,
                                        a_t: batch_a, r_t: batch_r})
                loss_epoch = loss_epoch + temp_loss
            self.GLOABL_LOSS_VECTOR.append((self.GLOBAL_IND_EPOCH, loss_epoch/batch_ind)) # append mean-error
            print('at epoch %d, loss=%f' % self.GLOABL_LOSS_VECTOR[-1])
            self.GLOBAL_IND_EPOCH = self.GLOBAL_IND_EPOCH + 1

    
    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(NONE_STATE)  # state=[0,0,0,0]
                self.train_queue[4].append(0.)  # 0 means terminal
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def predict(self, s_p, s_v):
        with self.default_graph.as_default():
            p, v = self.model.predict([s_p, s_v])
            return p, v

    def predict_p(self, s_p, s_v):
        with self.default_graph.as_default():
            p, v = self.model.predict([s_p, s_v])
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            s_p = s[:,:NUM_PRICE_STATE]
            s_p = np.expand_dims(s_p, -1)
            s_v = s[:,NUM_PRICE_STATE:]
            s_v = np.expand_dims(s_v, -1)
            p, v = self.model.predict([s_p, s_v])
            return v

#---------
frames = 0


class Agent:
    def __init__(self, eps_start, eps_end, eps_steps):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps

        self.memory = []  # used for n_step return
        self.R = 0.

    def getEpsilon(self):
        if(frames >= self.eps_steps):
            return self.eps_end
        else:
            # linearly interpolate
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps

    def act(self, s):
        eps = self.getEpsilon()
        global frames
        frames = frames + 1

        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)

        else:
            s = np.array([s])
            s1 = np.expand_dims(s, -1)
            s_p = s1[:,0:NUM_VOL_STATE,:]
            s_v = s1[:,NUM_VOL_STATE:,:]
            p = brain.predict_p(s_p, s_v)[0]

            # a = np.argmax(p)
            # randomly choose action based on the probabilities
            a = np.random.choice(NUM_ACTIONS, p=p)

            return a

    def train(self, s, a, r, s_):  # this is not training, instead, it just send the samples to the brain
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            # only take the last state after n-steps
            _, _, _, s_ = memory[n - 1]

            return s, a, self.R, s_  # R is the accumulated reward in the n steps

        # turn action into one-hot representation
        a_cats = np.zeros(NUM_ACTIONS)
        a_cats[a] = 1

        self.memory.append((s, a_cats, r, s_))  # 1-step sample

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:  # if terminal state, put the samples in memory into a n-step sample
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, a, r, s_)  # push n-step sample

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_)

            # subtract the immediate reward of the earlest state
            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)  # remove the earlest element

    # possible edge case - if an episode ends in <N steps, the computation is incorrect



#-- main
from tensorflow.python.framework import ops
ops.reset_default_graph()
#env_test = Environment(render=False)
NUM_STATE = 120
NUM_ACTIONS = 3
NONE_STATE = np.zeros(NUM_STATE)

brain = Brain()  # brain is global in A3C


#global_loss = np.array(GLOABL_LOSS_VECTOR)
stop_signal = False

render = False
data_path = './data_SPY/SPY_1998_2010_v2_filled.csv'
env = MarketEnv(data_path)
agent = Agent(EPS_START, EPS_STOP, EPS_STEPS)

ind = 0
num_generations = NUM_GENERATIONS
#print('***************************')
while ind < num_generations:
    #print('MMMMMMMMMMMM Episode %d' % ind)
    s = env.reset()
    R = 0
    action_no = 0
    while True:
#        time.sleep(THREAD_DELAY)  # yield

        if render:
            env.render()
#        print('NNNNNNNNNNNNN action Number %d' % action_no)
        action_no = action_no+1
        a = agent.act(s)
#        print('before action &&&&&, action = %d' % a)
#        print('before action &&&&&, current position is %d' % env.current_position)
        s_, r, done, info = env.step(a) 
#        print('after action ^^^^^^^:')
#        print(env.boughts)
#        print('after action, immediate reward = %f' % r)

        if done:  # terminal state
            s_ = None
        
        agent.train(s, a, r, s_)

        s = s_
        R += r

        if done or stop_signal:
            if len(brain.train_queue[0]) > MIN_BATCH:
                brain.optimize()
            break

    print("Total R at epsiode %d is %f:" % (ind, R))
    brain.GLOBAL_EPISODE_REWARDS.append((ind, R))
    if ind > 10:
        if not (ind % SAVED_EPISODES):
            result_ep_rewards1 = np.array(brain.GLOBAL_EPISODE_REWARDS)        
            df_results1 = pd.DataFrame(data=result_ep_rewards1, columns=['episode_no','ep_reward'])
            print('saved intermediate results at %d episodes' % ind)
            df_results1.to_csv('./training_result.csv')
            model_path = './model/model_2.h5'
            brain.model.save_weights(model_path)
            print('At episode %d, saved model to : %s' % (ind, model_path))
        
#    print('END of  Episode %d' % ind)
#    if R > 10.0:
#        import sys
#        print('abnomal episode reward')
##        print('long stop loss')
##        print(env.long_stop_loss_price)
#        print('short stop loss')
#        print(env.short_stop_loss_price)      
#        sys.exit(1)
    ind = ind + 1
    

result_ep_rewards = np.array(brain.GLOBAL_EPISODE_REWARDS)

df_results = pd.DataFrame(data=result_ep_rewards, columns=['episode_no','ep_reward'])
df_results.to_csv('./training_result.csv')

import matplotlib.pyplot as plt
plt.plot(result_ep_rewards[:,1], 'k-', label='Episode reward')
plt.xlabel('Epoch')
plt.ylabel('Episode reward')
plt.show()