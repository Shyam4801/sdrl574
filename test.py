
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
import time
import cv2

nb_Action = 8
maxStepsPerEpisode = 10000
np.random.seed(0)

"""
@Lin: control frame rate here
"""
DISPLAY = True
FRAME_SLEEP_TIME = 0.3


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def accomplish_subgoal(goal, RL_agent, env, episodeSteps, goalExplain, actionExplain, actionMap):
    while not env.is_game_end() and not env.agent_reached_goal(goal) and episodeSteps <= maxStepsPerEpisode:
        if DISPLAY:
            env.gym_env.render()
            time.sleep(FRAME_SLEEP_TIME)
        """
        @Lin: the state here is image observation
        """
        state = env.stack_states_together()
        """
        @Lin: let the RL agent to select low-level action
        """
        action = RL_agent.selectMove(state, goal)

        print("[Subgoal: {0}] episode step: {1}, action chosen: {2}".format(goalExplain[goal], episodeSteps, actionExplain[action]))

        externalRewards = env.act(actionMap[action])
        episodeSteps += 1
        nextState = env.stack_states_together()

    return episodeSteps


def main():
    """
    @Lin: this defines the low-level actions (output by RL)
    """
    actionMap = [0, 1, 2, 3, 4, 5, 11, 12]
    actionExplain = ['no action', 'jump', 'up', 'right', 'left', 'down', 'jump right', 'jump left']

    ############## -- DML modified -- ###########
    """
    @Lin: this define the sub-goals (output by Planner and taken as an input by the RL agent)
    """
    goalExplain = ['lower right ladder', 'jump to the left of devil', 'key', 'lower left ladder',
                   'lower right ladder', 'central high platform', 'right door']  # 7
    ################# -- end -- ###########

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="montezuma_revenge.bin")
    parser.add_argument("--display_screen", type=str2bool, default=True)
    parser.add_argument("--frame_skip", default=4)
    parser.add_argument("--color_averaging", default=True)
    parser.add_argument("--random_seed", default=0)
    parser.add_argument("--minimal_action_set", default=False)
    parser.add_argument("--screen_width", default=84)
    parser.add_argument("--screen_height", default=84)
    parser.add_argument("--load_weight", default=False)
    parser.add_argument("--use_sparse_reward", type=str2bool, default=True)
    args = parser.parse_args()
    env = ALEEnvironment(args.game, args)

    """
    @Lin: Load trained model (there are 7 pre-trained models)
    """
    # Initialize network and agent
    episodeCount = 0

    firstNet = Net()
    secondNet = Net()
    thirdNet = Net()
    fourthNet = Net()
    fifthNet = Net()
    sixthNet = Net()
    seventhNet = Net()
    firstNet.loadWeight(0)
    secondNet.loadWeight(1)
    thirdNet.loadWeight(2)
    fourthNet.loadWeight(3)
    fifthNet.loadWeight(4)
    sixthNet.loadWeight(5)
    seventhNet.loadWeight(6)

    RL_policies = [firstNet, secondNet, thirdNet, fourthNet, fifthNet, sixthNet, seventhNet]

    """
    @Lin: run 1 episodes
    """
    # for episode in range(80000):
    while episodeCount < 1:
        print("\n\n### EPISODE "  + str(episodeCount) + "###")
        # Restart the game
        """
        @Lin: they wrap the original Gym environment, the start_new_game() function is equivalent to the reset() function
        """
        env.start_new_game()
        episodeSteps = 0

        """
        @Lin: The interaction loop: run until current trajectory contains more than maxStepsPerEpisode steps 
            or the agent finishes the task
        """
        while not env.is_game_end() and episodeSteps <= maxStepsPerEpisode:
            stateLastGoal = env.stack_states_together()
            """
            @Lin: in normal cases, the subgoal here should be chosen by the planner (e.g. subgoal = planner(stateLastGoal))
                , but here they simply use hard-coded subgoals
            """
            for subgoal in [0, 1, 2, 3, 4, 5, 6]:
                print('predicted sub-goal is: ' + goalExplain[subgoal])
                episodeSteps = accomplish_subgoal(subgoal, RL_policies[subgoal], env, episodeSteps, goalExplain, actionExplain, actionMap)

                # Update subgoal
                if episodeSteps > maxStepsPerEpisode:
                    break
                elif env.agent_reached_goal(subgoal):
                    print('subgoal reached: ' + goalExplain[subgoal])
                else:
                    break
        episodeCount += 1


if __name__ == "__main__":
    main()
