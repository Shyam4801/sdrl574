# Hoang M. Le
# California Institute of Technology
# hmle@caltech.edu
# 
# Simple testing of trained subgoal models
# ===================================================================================================================

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
from controller_net import Net
from PIL import Image
from tensorboard import TensorboardVisualizer
from os import path
import planner
import math, random
from hyperparameters_new import *
from evaluate_plan import *
import pickle


def pause():
    os.system('read -s -n 1 -p "Press any key to continue...\n"')



def main():
    visualizer = TensorboardVisualizer()
    logdir = path.join(recordFolder + '/')  ## subject to change
    visualizer.initialize(logdir, None)

    # tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

    actionMap = [0, 1, 2, 3, 4, 5, 11, 12]

    actionExplain = ['no action', 'jump', 'up', 'right', 'left', 'down', 'jump right', 'jump left']

    goalExplain = ['lower right ladder', 'jump to the left of devil', 'key', 'lower left ladder',
                   'lower right ladder', 'central high platform', 'right door']  # 7

    Num_subgoal = len(goalExplain)
    subgoal_success_tracker = [[] for i in range(Num_subgoal)]  # corresponds to the 7 subgoals
    subgoal_trailing_performance = [0, 0, 0, 0, 0, 0, 0]  # corresponds to the 7 subgoals
    random_experience = [deque(), deque(), deque(), deque(), deque(), deque(), deque()]
    kickoff_lowlevel_training = [False, False, False, False, False, False, False]

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="montezuma_revenge.bin")
    parser.add_argument("--display_screen", type=str2bool, default=False)
    parser.add_argument("--frame_skip", default=4)
    parser.add_argument("--color_averaging", default=True)
    parser.add_argument("--random_seed", default=0)
    parser.add_argument("--minimal_action_set", default=False)
    parser.add_argument("--screen_width", default=84)
    parser.add_argument("--screen_height", default=84)
    parser.add_argument("--load_weight", default=False)
    parser.add_argument("--use_sparse_reward", type=str2bool, default=True)
    args = parser.parse_args()
    ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])
    MetaExperience = namedtuple("MetaExperience", ["state", "goal", "reward", "next_state", "done"])
    annealComplete = False
    saveExternalRewardScreen = True

    env = ALEEnvironment(args.game, args)
    # print "agent loc:",env.agent_location(env.getScreenRGB())

    # Initilize network and agent

    hdqn = Hdqn(GPU)
    hdqn1 = Hdqn(GPU)
    hdqn2 = Hdqn(GPU)
    hdqn3 = Hdqn(GPU)
    hdqn4 = Hdqn(GPU)
    hdqn5 = Hdqn(GPU)
    hdqn6 = Hdqn(GPU)

    hdqn_list = [hdqn, hdqn1, hdqn2, hdqn3, hdqn4, hdqn5, hdqn6]

    Num_hdqn = len(hdqn_list)  # 7 subgoal

    # for i in range(Num_hdqn):
    #     if i not in goal_to_train:
    #         hdqn_list[i].loadWeight(i) # load the pre-trained weights for subgoals that are not learned?
    #         kickoff_lowlevel_training[i] = True # switch this off

    agent = Agent(hdqn, range(nb_Action), range(Num_subgoal), defaultNSample=BATCH, defaultRandomPlaySteps=1000,
                  controllerMemCap=EXP_MEMORY, explorationSteps=50000, trainFreq=TRAIN_FREQ, hard_update=1000)
    agent1 = Agent(hdqn1, range(nb_Action), range(Num_subgoal), defaultNSample=BATCH, defaultRandomPlaySteps=20000,
                   controllerMemCap=EXP_MEMORY, explorationSteps=200000, trainFreq=TRAIN_FREQ,
                   hard_update=HARD_UPDATE_FREQUENCY)
    agent2 = Agent(hdqn2, range(nb_Action), range(Num_subgoal), defaultNSample=BATCH, defaultRandomPlaySteps=20000,
                   controllerMemCap=EXP_MEMORY, explorationSteps=200000, trainFreq=TRAIN_FREQ,
                   hard_update=HARD_UPDATE_FREQUENCY)
    agent3 = Agent(hdqn3, range(nb_Action), range(Num_subgoal), defaultNSample=BATCH, defaultRandomPlaySteps=20000,
                   controllerMemCap=EXP_MEMORY, explorationSteps=200000, trainFreq=TRAIN_FREQ,
                   hard_update=HARD_UPDATE_FREQUENCY)
    agent4 = Agent(hdqn4, range(nb_Action), range(Num_subgoal), defaultNSample=BATCH, defaultRandomPlaySteps=20000,
                   controllerMemCap=EXP_MEMORY, explorationSteps=200000, trainFreq=TRAIN_FREQ,
                   hard_update=HARD_UPDATE_FREQUENCY)
    agent5 = Agent(hdqn5, range(nb_Action), range(Num_subgoal), defaultNSample=BATCH, defaultRandomPlaySteps=20000,
                   controllerMemCap=EXP_MEMORY, explorationSteps=200000, trainFreq=TRAIN_FREQ,
                   hard_update=HARD_UPDATE_FREQUENCY)
    agent6 = Agent(hdqn6, range(nb_Action), range(Num_subgoal), defaultNSample=BATCH, defaultRandomPlaySteps=20000,
                   controllerMemCap=EXP_MEMORY, explorationSteps=200000, trainFreq=TRAIN_FREQ,
                   hard_update=HARD_UPDATE_FREQUENCY)
    # agent7 = Agent(hdqn7, range(nb_Action), range(Num_subgoal), defaultNSample=BATCH, defaultRandomPlaySteps=20000, controllerMemCap=EXP_MEMORY, explorationSteps=200000, trainFreq=TRAIN_FREQ,hard_update=HARD_UPDATE_FREQUENCY)
    # agent8 = Agent(hdqn7, range(nb_Action), range(Num_subgoal), defaultNSample=BATCH, defaultRandomPlaySteps=20000, controllerMemCap=EXP_MEMORY, explorationSteps=200000, trainFreq=TRAIN_FREQ,hard_update=HARD_UPDATE_FREQUENCY)
    agent_list = [agent, agent1, agent2, agent3, agent4, agent5, agent6]

    for i in range(Num_hdqn):
        # if i in goal_to_train:
        agent_list[i].compile()
        if i not in goal_to_train:
            agent_list[i].randomPlay = False
            agent_list[i].controllerEpsilon = 0.0

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

    # for episode in range(80000):
    record = []

    episodeCount = 0

    plantrace = []
    ro_table_lp = []

    nS = 14  # 6 locations, doubled with key picked, in total, 8 states, 1 good terminal (-2), 1 bad terminate (-3)
    nA = 6  # move to right ladder, move to key, move to left ladder, move to door, move to left of devil, move to initial

    R_table = np.zeros((nS, nA))
    ro_table = np.zeros((nS, nA))
    explore = True
    converged = False
    generate_goal_file(0)
    planabandoned = False

    cleanupconstraint()
    while episodeCount < EPISODE_LIMIT and stepCount < STEPS_LIMIT:
        print("\n\n### EPISODE " + str(episodeCount) + "###")
        # Restart the game
        env.start_new_game()
        episodeSteps = 0

        replanned = False
        stateaction = []
        planquality = 0
        generate_rovalue_from_table(env, ro_table_lp, ro_table)

        done = False
        allsubgoallearned = True

        if explore:
            print("generate new plan...")
            oldplan = plantrace
            plantrace = generateplan() #PLAN TRACE:   [(1, 'move(lower_right_ladder) ', 'at(plat1) cost(0) '), (2, '', 'at(lower_right_ladder) cost(50) ')]
            planabandoned = False
            if plantrace == None:
                #    print "Run",run
                print("No plan found at Episode", episodeCount)
                print("I think the symbolic plan has converged, so I will continue executing the same plan now.")
                converged = True
                plantrace = oldplan
        if not explore:
            print("continue executing previous plan...")
            done = False

        # Run episode
        goal_index = 0
        goal_not_found = False

        # dispatch each subgoal to DQN
        # goal_index denotes the current index of action/symbolic transition to dispatch
        while not env.is_game_end() and episodeSteps <= maxStepsPerEpisode and goal_index < len(
                plantrace) - 1 and not goal_not_found:

            goal = selectSubGoal(plantrace, goal_index)
            if not option_learned[goal]:
                allsubgoallearned = False
            state_ind, action_ind = obtainStateAction(plantrace, goal_index)
            if goal == -1:
                print("Subgoal not found for ", plantrace[goal_index + 1][2])
                # now tell the planenr that don't generate such unpromising actions, by punishing with a big reward.
                # shortly we will have DQN training to rule those bad actions out.
                goal_not_found = True
            else:  # goal found
                print('current state and action:', plantrace[goal_index][2], state_ind, plantrace[goal_index][2],
                      action_ind)
                print('predicted subgoal is: ', plantrace[goal_index + 1][2])
                print('goal explain', goalExplain[goal])
                #    pause()
                # pretrained neural network perform execution.
                # This part can be extended into execution with learning
                loss_list = []
                avgQ_list = []
                tdError_list = []
                planabandoned = False

                # train DQN for the subgoal
                while not env.is_game_end() and not env.agent_reached_goal(goal) and episodeSteps <= maxStepsPerEpisode:

                    state = env.stack_states_together()
                    # action = agent_list[goal].selectMove(state, goal)
                    action = agent_list[goal].selectMove(state)
                    externalRewards = env.act(actionMap[action])

                    # stepCount += 1
                    episodeSteps += 1
                    nextState = env.stack_states_together()

                    # only assign intrinsic reward if the goal is reached and it has not been reached previously
                    intrinsicRewards = agent_list[goal].criticize(env.agent_reached_goal(goal), actionMap[action],
                                                                  env.is_game_end(), 0, args.use_sparse_reward)
                    # Store transition and update network params, only when it is not learnted yet
                    if not option_learned[goal]:
                        if agent_list[goal].randomPlay:
                            exp = ActorExperience(state, goal, action, intrinsicRewards, nextState, env.is_game_end())
                            random_experience[goal].append(exp)
                            if len(random_experience[goal]) > 20000:
                                random_experience[goal].popleft()
                        else:
                            if not kickoff_lowlevel_training[goal]:
                                print("not kick off low level training yet")
                                for exp in random_experience[goal]:
                                    agent_list[goal].store(exp)
                                    option_t[goal] += 1
                                    option_training_counter[goal] += 1
                                print("Finally, the number of stuff in random_experience is", len(random_experience[goal]))
                                print("The number of item in experience memory so far is:", len(agent_list[goal].memory))
                                random_experience[goal].clear()
                                assert len(random_experience[goal]) == 0
                                kickoff_lowlevel_training[goal] = True
                                print("This should really be one time thing")
                                print(" number of option_t is ", option_t[goal])
                            else:
                                if not option_learned[goal]:
                                    exp = ActorExperience(state, goal, action, intrinsicRewards, nextState,
                                                          env.is_game_end())
                                    agent_list[goal].store(exp)
                                    option_t[goal] += 1
                                    option_training_counter[goal] += 1

                        if (option_t[goal] >= agent_list[goal].defaultRandomPlaySteps) and (
                        not agent_list[goal].randomPlay):
                            if (option_t[goal] == agent_list[goal].defaultRandomPlaySteps):
                                print('start training (random walk ends) for subgoal ' + str(goal))

                            if (option_t[goal] % agent_list[goal].trainFreq == 0 and option_training_counter[
                                goal] > 0 and (not option_learned[goal])):
                                loss, avgQ, avgTDError = agent_list[goal].update(option_t[goal])
                                print("Perform training on experience replay")
                                print("loss:", loss, "avgQ:", avgQ, "avgTDError", avgTDError)

                                loss_list.append(loss)
                                avgQ_list.append(avgQ)
                                tdError_list.append(avgTDError)
                                option_training_counter[goal] = 0

            stateaction.append((state_ind, action_ind))
            if (state_ind, action_ind) not in ro_table_lp:
                ro_table_lp.append((state_ind, action_ind))

            # train meta-controller using R learning
            if goal_not_found:
                print('Untrainable symbolic actions.')
                reward = -200
                state_next = -3
                R_table[state_ind, action_ind] += 0.1 * (
                            reward - ro_table[state_ind, action_ind] + max(R_table[state_next, :]) - R_table[
                        state_ind, action_ind])
                ro_table[state_ind, action_ind] += 0.5 * (
                            reward + max(R_table[state_next, :]) - max(R_table[state_ind, :]) - ro_table[
                        state_ind, action_ind])
                print('R(', state_ind, action_ind, ')=', R_table[state_ind, action_ind])
                print('ro(', state_ind, action_ind, ')=', ro_table[state_ind, action_ind])
                updateconstraint(state_ind, action_ind)
                planabandoned = True
                break
            elif (episodeSteps > maxStepsPerEpisode) or env.is_game_end():
                # failed plan, receive intrinsic reward of -100
                print('Goal not achieved.')
                subgoal_success_tracker[goal].append(0)
                faluretimes = subgoal_success_tracker[goal].count(0)
                print('Failure times:', subgoal_success_tracker[goal].count(0))
                state_next = -3
                if not option_learned[goal]:
                    if faluretimes > 10000:
                        if goal not in bad_option:
                            bad_option.append(goal)
                        print("abandoned options:", bad_option)
                        updateconstraint(state_ind, action_ind)
                        planabandoned = True
                        reward = -200
                    else:
                        reward = -10  # - subgoal_success_tracker[goal].count(0)
                else:
                    reward = -10

                R_table[state_ind, action_ind] += 0.1 * (
                            reward - ro_table[state_ind, action_ind] + max(R_table[state_next, :]) - R_table[
                        state_ind, action_ind])
                ro_table[state_ind, action_ind] += 0.5 * (
                            reward + max(R_table[state_next, :]) - max(R_table[state_ind, :]) - ro_table[
                        state_ind, action_ind])
                print('R(', state_ind, action_ind, ')=', R_table[state_ind, action_ind])
                print('ro(', state_ind, action_ind, ')=', ro_table[state_ind, action_ind])
                break
            elif env.agent_reached_goal(goal):
                subgoal_success_tracker[goal].append(1)
                goalstate = plantrace[goal_index + 1][2]
                previousstate = plantrace[goal_index][2]
                print('previous state', previousstate)
                print('goal reached', goalstate)
                print('Success times:', subgoal_success_tracker[goal].count(1))
                #    print 'current state:', env.stack_states_together()
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

                print('R(', state_ind, action_ind, ')=', R_table[state_ind, action_ind])
                print('ro(', state_ind, action_ind, ')=', ro_table[state_ind, action_ind])

                if not option_learned[goal]:
                    if agent_list[goal].randomPlay:
                        agent_list[goal].randomPlay = False
                        # option_t[goal] = 0 ## Reset option counter
                    episodeSteps = 0  ## reset episode steps to give new goal all 500 steps
                    print("now setting episodeSteps to be", episodeSteps)
                #    print "randomPlay:",agent_list[goal].randomPlay

                # time.sleep(60)
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
            print("An action in this plan is abandoned. Exploration must start")
            explore = True
        elif not allsubgoallearned:
            print("trying to train subgoal DQN. Continue executing the same plan")
            explore = False
        else:
            eps = 0.2
            explore = (throwdice(eps) and not converged) or replanned

        if explore:
            generate_goal_file(planquality)
        episodeCount += 1

        for subgoal in goal_to_train:
            if len(subgoal_success_tracker[subgoal]) > 100:
                subgoal_trailing_performance[subgoal] = sum(subgoal_success_tracker[subgoal][-100:]) / 100.0
                if subgoal_trailing_performance[subgoal] > STOP_TRAINING_THRESHOLD:
                    if not option_learned[subgoal]:
                        option_learned[subgoal] = True
                        hdqn_list[subgoal].saveWeight(subgoal)
                        time.sleep(60)
                        agent_list[subgoal].clear_memory(subgoal)
                        hdqn_list[subgoal].clear_memory()
                        print("Training completed after for subgoal", subgoal, "Model saved")
                    #    if subgoal == (nb_Option-1):
                    #        training_completed = True ## Stop training, all done
                    else:
                        print("Subgoal ", subgoal, " should no longer be in training")
                elif subgoal_trailing_performance[subgoal] < STOP_TRAINING_THRESHOLD and option_learned[subgoal]:
                    print("For some reason, the performance of subgoal ", subgoal, " dropped below the threshold again")
                    if subgoal_trailing_performance[subgoal] == 0.:
                        option_learned[subgoal] = False
            else:
                subgoal_trailing_performance[subgoal] = 0.0
            print("Trailing success ratio for " + str(subgoal) + " is:", subgoal_trailing_performance[subgoal])

        if (not annealComplete):
            # Annealing
            print("perform annealing")
            for subgoal in goal_to_train:
                agent_list[subgoal].annealControllerEpsilon(option_t[subgoal], option_learned[subgoal])

        stepCount = sum(option_t)

        if stepCount > 10:  ## Start plotting after certain number of steps
            for subgoal in goal_to_train:
                visualizer.add_entry(option_t[subgoal], "trailing success ratio for goal " + str(subgoal),
                                     subgoal_trailing_performance[subgoal])
            visualizer.add_entry(stepCount, "average Q values", np.mean(avgQ_list))
            visualizer.add_entry(stepCount, "training loss", np.mean(loss_list))
            visualizer.add_entry(stepCount, "average TD error", np.mean(tdError_list))


if __name__ == "__main__":
    main()
