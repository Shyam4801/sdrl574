from config.hyperparameters import *
from utils.replay_buffer import PrioritizedReplayBuffer
from utils.schedules import LinearSchedule
from keras.models import Model, Sequential
from keras.layers import Dense, Input, concatenate, Lambda, Conv2D, Flatten
from keras import optimizers
from keras import initializers
import gc
from constants import *
from loss import clipped_masked_error

beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                               initial_p=prioritized_replay_beta0,
                               final_p=1.0)

class DDQNAgent:
    def __init__(self, actions_size, state_size, defaultNSample, defaultRandomPlaySteps, controllerMemCap, explorationSteps, trainFreq, hard_update,
                 controllerEpsilon=defaultControllerEpsilon):
        
        # Initialize atributes
        self._state_size = state_size
        self._action_size = actions_size
        self.controllerEpsilon = controllerEpsilon
        self.nSamples = defaultNSample 
        self.gamma = defaultGamma
        self.memory = PrioritizedReplayBuffer(controllerMemCap, alpha=prioritized_replay_alpha)
        self.enable_double_dqn = True
        self.exploration = LinearSchedule(schedule_timesteps = explorationSteps, initial_p = 1.0, final_p = 0.02)
        self.defaultRandomPlaySteps = defaultRandomPlaySteps
        self.trainFreq = trainFreq
        self.randomPlay = True
        self.learning_done = False
        self.hard_update = hard_update
        self.enable_dueling_network = False

        rmsProp = optimizers.RMSprop(lr=LEARNING_RATE, rho=0.95, epsilon=1e-08, decay=0.0)
        # Build networks
        self.primary_network = self._build_network()
        self.target_network = self._build_network()
        self.enable_dueling(self.enable_dueling_network,self.primary_network,self.target_network)
        
        self.primary_network.compile(loss='mse', optimizer=rmsProp)           
        self.target_network.compile(loss='mse', optimizer=rmsProp)
        
   
    def _build_network(self):
        network = Sequential()
        network.add(Conv2D(32, (8, 8), strides=4, activation='relu', padding='valid', input_shape=(84, 84, 4)))
        network.add(Conv2D(64, (4, 4), strides=2, activation='relu', padding='valid'))
        network.add(Conv2D(64, (4, 4), strides=2, activation='relu', padding='valid'))
        network.add(Flatten())
        network.add(Dense(HIDDEN_NODES, activation='relu',
                                kernel_initializer=initializers.random_normal(stddev=0.01, seed=SEED)))
        network.add(Dense(len(self._action_size), activation='linear',
                                kernel_initializer=initializers.random_normal(stddev=0.01, seed=SEED)))
        
        return network
    
    def enable_dueling(self, enable_dueling_network, net1, net2):
        if not self.enable_dueling_network:
            self.controllerNet = net1
            self.targetControllerNet = net2

            self.controllerNet.reset_states()
            self.targetControllerNet.set_weights(self.controllerNet.get_weights())
            self.primary_network = self.controllerNet
            self.target_network = self.targetControllerNet
        else:
            layer = net1.layers[-2]
            nb_output = net1.output._keras_shape[-1]
            y = Dense(nb_output + 1, activation='linear', kernel_initializer=initializers.random_normal(stddev=0.01))(
                layer.output)
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                                 output_shape=(nb_output,))(y)

            self.controllerNet = Model(inputs=net1.input, outputs=outputlayer)
            self.primary_network = self.controllerNet
            self.controllerNet.compile(optimizer=rmsProp, loss='mse')
            self.targetControllerNet = clone_model(self.controllerNet)
            self.target_network = self.targetControllerNet
            self.targetControllerNet.compile(optimizer=rmsProp, loss='mse')


    # Select a random action or predict next action
    def act(self, state):
        if not self.learning_done:
            if self.controllerEpsilon < random.random():
                return np.argmax(self.primary_network.predict([np.reshape(state, (1, 84, 84, 4))], verbose=0))
            return random.choice(self._action_size) # choose a random action based on the current state
        else:
            return np.argmax(self.simple_net.predict([np.reshape(state, (1, 84, 84, 4))], verbose=0))


    def criticize(self, reachGoal, action, die, reward_from_distance, useSparseReward):
        reward = 0.0
        if reachGoal:
            reward += 1.0
        if die:
            reward -= 1.0
        if not useSparseReward:
            reward += reward_from_distance
        reward = np.minimum(reward, maxReward)
        reward = np.maximum(reward, minReward)
        return reward

    # Store <current state, action, reward, next_state, status> in replay buffer
    def store(self, experience):
        self.memory.add(experience.state, experience.action, experience.reward, experience.next_state, experience.done)

    def compile(self):
        y_pred = self.primary_network.output
        y_true = Input(name='y_true', shape=(nb_Action,))
        mask = Input(name='mask', shape=(nb_Action,))
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_pred, y_true, mask])
        ins = [self.primary_network.input] if type(self.primary_network.input) is not list else self.primary_network.input
        trainable_model = Model(inputs=ins + [y_true, mask], outputs=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2
        losses = [
            lambda y_true, y_pred: y_pred,  
            lambda y_true, y_pred: K.zeros_like(y_pred),  
        ]
        rmsProp = optimizers.RMSprop(lr=LEARNING_RATE, rho=0.95, epsilon=1e-08, decay=0.0)
        trainable_model.compile(optimizer=rmsProp, loss=losses)
        self.trainable_model = trainable_model
        self.compiled = True

    def _update(self, stepCount):
        batches = self.memory.sample(self.nSamples, beta=beta_schedule.value(stepCount))
        (stateVector, actionVector, rewardVector, nextStateVector, doneVector, importanceVector, idxVector) = batches
        
        stateVector = np.asarray(stateVector)
        nextStateVector = np.asarray(nextStateVector)
        
        q_values = self.primary_network.predict(stateVector)
        assert q_values.shape == (self.nSamples, nb_Action)
        # Double DQN to ensure that the values are not overestimated
        if self.enable_double_dqn:
            actions = np.argmax(q_values, axis = 1)
            assert actions.shape == (self.nSamples,)

            target_q_values = self.target_network.predict(nextStateVector)
            assert target_q_values.shape == (self.nSamples, nb_Action)
            q_batch = target_q_values[range(self.nSamples), actions]        # using an estimate of the q values instead of max 
            assert q_batch.shape == (self.nSamples,)
        else:
            target_q_values = self.target_network.predict(nextStateVector)
            q_batch = np.max(target_q_values, axis=1)             # using max q to update the best Q
            assert q_batch.shape == (self.nSamples,)

        targets = np.zeros((self.nSamples, nb_Action))
        dummy_targets = np.zeros((self.nSamples,))
        masks = np.zeros((self.nSamples, nb_Action))

        # Compute reward + gamma * max_action Q(nextState, action) and update the target targets accordingly
        discounted_reward_batch = self.gamma * q_batch
        # Set discounted reward to zero for all states that were terminal.
        terminalBatch = np.array([1-float(done) for done in doneVector])
        assert terminalBatch.shape == (self.nSamples,)
        discounted_reward_batch *= terminalBatch
        reward_batch = np.array(rewardVector)
        action_batch = np.array(actionVector)
        assert discounted_reward_batch.shape == reward_batch.shape
        Rs = reward_batch + discounted_reward_batch
        for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
            target[action] = R  # update action with accumulated reward estimated
            dummy_targets[idx] = R
            mask[action] = 1.  # enable loss for this specific action
        td_errors = targets[range(self.nSamples), action_batch] - q_values[range(self.nSamples), action_batch]
        
        new_priorities = np.abs(td_errors) + prioritized_replay_eps
        self.memory.update_priorities(idxVector, new_priorities)
        
        targets = np.array(targets).astype('float32')
        masks = np.array(masks).astype('float32')

        ins = [stateVector] if type(self.primary_network.input) is not list else stateVector
        if stepCount >= self.defaultRandomPlaySteps:
            loss = self.trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets], sample_weight = [np.array(importanceVector), np.ones(self.nSamples)])
        else:
            loss = [0.0,0.0,0.0]
        
        if stepCount > self.defaultRandomPlaySteps and stepCount % self.hard_update == 0:
            self.target_network.set_weights(self.primary_network.get_weights())
        return loss[1], np.mean(q_values), np.mean(np.abs(td_errors))
        

    def update(self, stepCount):
        loss = self._update(stepCount)
        return loss

    def annealControllerEpsilon(self, stepCount, option_learned):
        if not self.randomPlay:
            if option_learned:
                self.controllerEpsilon = 0.0
            else:
                if stepCount > self.defaultRandomPlaySteps:
                    self.controllerEpsilon = self.exploration.value(stepCount - self.defaultRandomPlaySteps)

    def clear_memory(self, goal):
        self.learning_done = True 
        del self.trainable_model
        del self.memory
        gpu = self.net.gpu
        del self.net
        gc.collect()

        rmsProp = optimizers.RMSprop(lr=LEARNING_RATE, rho=0.95, epsilon=1e-08, decay=0.0)
        self.simple_net = self._build_network()
        self.simple_net.compile(loss = 'mse', optimizer = rmsProp)
        self.simple_net.load_weights(recordFolder+'/policy_subgoal_' + str(goal) + '.h5')
        self.simple_net.reset_states()

