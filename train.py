import os

os.environ['PYTHONHASHSEED'] = '0'

import argparse
import sys
import time
import numpy as np
import tensorflow as tf
from collections import namedtuple, deque
from environment import ALEEnvironment
from agent import Agent
from hdqn import Hdqn
from PIL import Image
from utils.tensorboard import TensorboardVisualizer
from os import path
import planner
import math, random
from config.hyperparameters import *
from evaluate_plan import *
import pickle
from constants import *

def main():
    # Visualizer
    visualizer = TensorboardVisualizer()
    logdir = path.join(recordFolder + '/')  
    visualizer.initialize(logdir, None)


    # We are focusing on 7 sub goals from the authors set of 13 subgoals
    Num_subgoal = len(goalExplain)
    subgoal_success_tracker = [[] for i in range(Num_subgoal)]  
    subgoal_training_performance = [0, 0, 0, 0, 0, 0, 0] 
    random_experience = [deque(), deque(), deque(), deque(), deque(), deque(), deque()]
    start_training = [False, False, False, False, False, False, False]

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])
    MetaExperience = namedtuple("MetaExperience", ["state", "goal", "reward", "next_state", "done"])
    annealComplete = False
    saveExternalRewardScreen = True

    # Intialize the atari environment
    env = ALEEnvironment()

    # Initialize network and agents
    hdqn_dict = {}
    agent_dict = {}
    for i in range(7):
        hdqn_dict[i] = Hdqn()
    
    num_hdqns = len(hdqn_dict) 

    agent_dict[0] = Agent(hdqn_dict[0], range(nb_Action), range(Num_subgoal), defaultNSample=BATCH, defaultRandomPlaySteps=1000,
                  controllerMemCap=EXP_MEMORY, explorationSteps=50000, trainFreq=TRAIN_FREQ, hard_update=1000)
    print(agent_dict[0])
    for i in range(1,7):
        agent_dict[i] = Agent(hdqn_dict[i], range(nb_Action), range(Num_subgoal), defaultNSample=BATCH, defaultRandomPlaySteps=20000,
                   controllerMemCap=EXP_MEMORY, explorationSteps=200000, trainFreq=TRAIN_FREQ,
                   hard_update=HARD_UPDATE_FREQUENCY)
    
    for i in range(num_hdqns):
        agent_dict[i].compile()
        if i not in goal_to_train:
            agent_dict[i].randomPlay = False
            agent_dict[i].controllerEpsilon = 0.0

    option_learned = [False, False, False, False, False, False, False]
    training_completed = False

    for i in range(Num_subgoal):
        if i not in goal_to_train:
            option_learned[i] = True

    episodeCount = 0
    stepCount = 0

    option_t = [0, 0, 0, 0, 0, 0, 0]
    option_training_counter = [0, 0, 0, 0, 0, 0, 0]
    bad_option = []
    record = []

    episodeCount = 0

    plantrace = []
    ro_table_lp = []

    nS = 14  # 6 locations, with/without key
    nA = 6  # move to right ladder, move to left of devil, move to left ladder, move to key, move to middle platform, move to door

    R_table = np.zeros((nS, nA)) # avg cumulative reward which is the environment reward used by the planner to update fluent rho
    ro_table = np.zeros((nS, nA)) # extrinsic reward from the environment 
    explore = True
    converged = False
    generate_goal_file(0)
    planabandoned = False
    cleanupconstraint()

    while episodeCount < EPISODE_LIMIT and stepCount < STEPS_LIMIT:
        print("\n\n######################## EPISODE " + str(episodeCount) + " ##############################")
        # Restart the game
        env.start_new_game()
        episodeSteps = 0

        replanned = False
        stateaction = []
        planquality = 0
        generate_rovalue(env, ro_table_lp, ro_table)

        done = False
        allsubgoallearned = True

        if explore:
            print("Generating new plan !")
            oldplan = plantrace
            plantrace = generateplan() 
            planabandoned = False
            if plantrace == None:
                print("No plan found !", "Episode: ", episodeCount)
                print("Perhaps symbolic plan has converged, executing the same plan ")
                converged = True
                plantrace = oldplan
        if not explore:
            print("Keep executing previous plan...")
            done = False

        # Run episode
        goal_index = 0
        goal_not_found = False

        # dispatch each subgoal to DQN
        # goal_index denotes the current index of symbolic transition
        while not env.is_game_end() and episodeSteps <= maxStepsPerEpisode and goal_index < len(
                plantrace) - 1 and not goal_not_found:

            goal = selectSubGoal(plantrace, goal_index)
            if not option_learned[goal]:
                allsubgoallearned = False
            state_ind, action_ind = obtainStateAction(plantrace, goal_index)
            if goal == -1:
                print("Subgoal not found for ", plantrace[goal_index + 1][2])
                # Penalize the action 
                goal_not_found = True
            else:  
                print('Current state and action:', plantrace[goal_index][2], state_ind, plantrace[goal_index][2],
                      action_ind)
                print('Predicted subgoal is: ', plantrace[goal_index + 1][2])
                print('Goal: ', goalExplain[goal])

                loss_list = []
                avgQ_list = []
                tdError_list = []
                planabandoned = False

                # Double DQN training for each subgoal
                while not env.is_game_end() and not env.agent_reached_goal(goal) and episodeSteps <= maxStepsPerEpisode:

                    state = env.stack_states_together() # 4D tensor
                    action = agent_dict[goal].selectMove(state)
                    externalRewards = env.act(actionMap[action])
                    episodeSteps += 1
                    nextState = env.stack_states_together()

                    # Assign intrinsic reward
                    intrinsicRewards = agent_dict[goal].criticize(env.agent_reached_goal(goal), actionMap[action],
                                                                  env.is_game_end(), 0, USE_SPARSE_REWARD)
                    # Store transition and update network parameters
                    if not option_learned[goal]:
                        if agent_dict[goal].randomPlay:
                            exp = ActorExperience(state, goal, action, intrinsicRewards, nextState, env.is_game_end())
                            random_experience[goal].append(exp)
                            if len(random_experience[goal]) > 20000:
                                random_experience[goal].popleft()
                        else:
                            if not start_training[goal]:
                                print("Not enough samples !")
                                for exp in random_experience[goal]:
                                    agent_dict[goal].store(exp)
                                    option_t[goal] += 1
                                    option_training_counter[goal] += 1
                                print("Samples in experience memory :", len(agent_dict[goal].memory))
                                random_experience[goal].clear()
                                assert len(random_experience[goal]) == 0
                                start_training[goal] = True
                            else:
                                if not option_learned[goal]:
                                    exp = ActorExperience(state, goal, action, intrinsicRewards, nextState,
                                                          env.is_game_end())
                                    agent_dict[goal].store(exp)
                                    option_t[goal] += 1
                                    option_training_counter[goal] += 1

                        if (option_t[goal] >= agent_dict[goal].defaultRandomPlaySteps) and (
                        not agent_dict[goal].randomPlay):
                            if (option_t[goal] == agent_dict[goal].defaultRandomPlaySteps):
                                print('Start training subgoal ' + str(goal))

                            if (option_t[goal] % agent_dict[goal].trainFreq == 0 and option_training_counter[
                                goal] > 0 and (not option_learned[goal])):
                                loss, avgQ, avgTDError = agent_dict[goal].update(option_t[goal])
                                print("Start training on experience replay")
                                print("loss:", loss, "avgQ:", avgQ, "avgTDError", avgTDError)

                                loss_list.append(loss)
                                avgQ_list.append(avgQ)
                                tdError_list.append(avgTDError)
                                option_training_counter[goal] = 0

            stateaction.append((state_ind, action_ind))
            if (state_ind, action_ind) not in ro_table_lp:
                ro_table_lp.append((state_ind, action_ind))

            # R learning [Meta Controller]
            if goal_not_found:
                print('Symbolic action is not trainable! ')
                reward = -200 # penalty
                state_next = -3
                R_table[state_ind, action_ind] += 0.1 * (
                            reward - ro_table[state_ind, action_ind] + max(R_table[state_next, :]) - R_table[
                        state_ind, action_ind])
                ro_table[state_ind, action_ind] += 0.5 * (
                            reward + max(R_table[state_next, :]) - max(R_table[state_ind, :]) - ro_table[
                        state_ind, action_ind])
                updateconstraint(state_ind, action_ind)
                planabandoned = True
                break
            elif (episodeSteps > maxStepsPerEpisode) or env.is_game_end():
                print('Goal not accomplished! ')
                subgoal_success_tracker[goal].append(0)
                faluretimes = subgoal_success_tracker[goal].count(0)
                print('Failed :', subgoal_success_tracker[goal].count(0),' times')
                state_next = -3
                if not option_learned[goal]:
                    if faluretimes > 10000:
                        if goal not in bad_option:
                            bad_option.append(goal)
                        print("Drop options !", bad_option)
                        updateconstraint(state_ind, action_ind)
                        planabandoned = True
                        reward = -200
                    else:
                        reward = -10  
                else:
                    reward = -10

                R_table[state_ind, action_ind] += 0.1 * (
                            reward - ro_table[state_ind, action_ind] + max(R_table[state_next, :]) - R_table[
                        state_ind, action_ind])
                ro_table[state_ind, action_ind] += 0.5 * (
                            reward + max(R_table[state_next, :]) - max(R_table[state_ind, :]) - ro_table[
                        state_ind, action_ind])
                break
            elif env.agent_reached_goal(goal):
                subgoal_success_tracker[goal].append(1)
                goalstate = plantrace[goal_index + 1][2]
                previousstate = plantrace[goal_index][2]
                print('Previous state', previousstate)
                print('Goal reached', goalstate)
                print('Successfully reached :', subgoal_success_tracker[goal].count(1), " times")
                if obtainedKey(previousstate, goalstate):
                    print("Obtained key! Get 100 reward!")
                    reward = 100
                elif openDoor(previousstate, goalstate):
                    print("Open the door! Get 300 reward!")
                    reward = 300
                    done = True
                else:
                    if not option_learned[goal]:
                        reward = 10
                    else:
                        reward = -10
                print(goal_index)
                if goal_index == len(plantrace) - 2:
                    state_next = -2
                else:
                    state_next = selectSubGoal(plantrace, goal_index + 1)

                R_table[state_ind, action_ind] += 0.1 * (
                            reward - ro_table[state_ind, action_ind] + max(R_table[state_next, :]) - R_table[
                        state_ind, action_ind])
                ro_table[state_ind, action_ind] += 0.5 * (
                            reward + max(R_table[state_next, :]) - max(R_table[state_ind, :]) - ro_table[
                        state_ind, action_ind])

                print('R-value[', state_ind, action_ind, '] =', R_table[state_ind, action_ind])
                print('ro_value[', state_ind, action_ind, '] =', ro_table[state_ind, action_ind])

                if not option_learned[goal]:
                    if agent_dict[goal].randomPlay:
                        agent_dict[goal].randomPlay = False
                    episodeSteps = 0  
                if done:
                    for i in range(15):
                        env.act(3)
                    for i in range(15):
                        env.act(0)
                    break
                goal_index += 1
            else:
                break
        planquality = calculateplanquality(ro_table, stateaction)
        print("plan quality is:", planquality)
        if planabandoned:
            print("An action abandoned. Explore !")
            explore = True
        elif not allsubgoallearned:
            print("Continue executing the same plan")
            explore = False
        else:
            eps = 0.2
            explore = (throwdice(eps) and not converged) or replanned

        if explore:
            generate_goal_file(planquality)
        episodeCount += 1

        for subgoal in goal_to_train:
            if len(subgoal_success_tracker[subgoal]) > 100:
                subgoal_training_performance[subgoal] = sum(subgoal_success_tracker[subgoal][-100:]) / 100.0
                if subgoal_training_performance[subgoal] > STOP_TRAINING_THRESHOLD:
                    if not option_learned[subgoal]:
                        option_learned[subgoal] = True
                        hdqns[subgoal].saveWeight(subgoal)
                        time.sleep(60)
                        agent_dict[subgoal].clear_memory(subgoal)
                        hdqns[subgoal].clear_memory()
                        print("Training completed for subgoal", subgoal)
                    else:
                        print("Subgoal ", subgoal, " should no longer be in training")
                elif subgoal_training_performance[subgoal] < STOP_TRAINING_THRESHOLD and option_learned[subgoal]:
                    print("Dropped below the threshold", subgoal)
                    if subgoal_training_performance[subgoal] == 0.:
                        option_learned[subgoal] = False
            else:
                subgoal_training_performance[subgoal] = 0.0
            print("Success ratio for " + str(subgoal) + ":", subgoal_training_performance[subgoal])

        if (not annealComplete):
            # Annealing
            for subgoal in goal_to_train:
                agent_dict[subgoal].annealControllerEpsilon(option_t[subgoal], option_learned[subgoal])

        stepCount = sum(option_t)

        if stepCount > 10:  
            for subgoal in goal_to_train:
                visualizer.add_entry(option_t[subgoal], "training success ratio for goal " + str(subgoal),
                                     subgoal_training_performance[subgoal])
            visualizer.add_entry(stepCount, "average Q values", np.mean(avgQ_list))
            visualizer.add_entry(stepCount, "training loss", np.mean(loss_list))
            visualizer.add_entry(stepCount, "average TD error", np.mean(tdError_list))


if __name__ == "__main__":
    main()
