
from config.hyperparameters import *
import sys
import os
import logging
import cv2
import numpy as np
from PIL import Image, ImageDraw
import gym
import os
import argparse

logger = logging.getLogger(__name__)
np.random.seed(SEED)


class ALEEnvironment:
    def __init__(self, rom_file, args):
        self.gym_env = gym.make("ALE/MontezumaRevenge-v5",render_mode="rgb_array")  #MontezumaRevenge-v0
        self.ale = self.gym_env.ale
        """
        @Lin: states for DRL agent are stacked consecutive frames, e.g. in the shape of 4*64*64
        """
        self.histLen = 4

        self.ale.setInt('frame_skip', args.frame_skip)
        self.ale.setFloat('repeat_action_probability', 0.0)
        self.ale.setBool('color_averaging', args.color_averaging)

        # if args.random_seed:
        #  self.ale.setInt('random_seed', args.random_seed)
        self.ale.setInt('random_seed', 0)  # hoang addition to fix the random seed across all environment

        if args.minimal_action_set:
            self.actions = self.ale.getMinimalActionSet()
            logger.info("Using minimal action set with size %d" % len(self.actions))
        else:
            self.actions = self.ale.getLegalActionSet()
            logger.info("Using full action set with size %d" % len(self.actions))
        logger.debug("Actions: " + str(self.actions))

        self.screen_width = args.screen_width
        self.screen_height = args.screen_height

        self.mode = "train"
        self.life_lost = False
        self.initSrcreen = self.getScreen()
        print("size of screen is:", self.initSrcreen.shape)
        im = Image.fromarray(self.initSrcreen)
        im.save('initial_screen.jpeg')

        """
        @Lin: here they use hard-coded sub-goals
        """
        """
        @Sid: need to find the goal location for both the agents
        """
        
        self.goalSet = []
        # goal 0
        self.goalSet.append([[69, 68], [73, 71]])  # Lower Right Ladder. This is the box for detecting first subgoal
        # self.goalSet.append([[11, 58], [15, 66]]) # lower left ladder 3
        # self.goalSet.append([[11, 68], [15, 71]])  # lower left ladder 3
        # goal 2
        self.goalSet.append([[7, 41], [11, 45]])  # Key. This will be second sub goal
        # goal 3
        self.goalSet.append([[11, 68], [15, 71]])  # lower left ladder 3
        # goal 4
        self.goalSet.append([[69, 68], [73, 71]])  # Lower Right Ladder again, this will be the third subgoal
        # goal 6
        self.goalSet.append([[70, 20], [73, 35]])  # Right Door. This will be the 4th subgoal
        self.goalCenterLoc = []
        
        
        for goal in self.goalSet:
            goalCenter = [float(goal[0][0] + goal[1][0]) / 2, float(goal[0][1] + goal[1][1]) / 2]
            self.goalCenterLoc.append(goalCenter)
        self.agent_original_location = [42, 33]
        self.agent_previous_X_coord = 42
        self.agent_previous_Y_coord = 33
        self.devil_previous_X_coord = 0
        self.devil_previous_Y_coord = 0
        self.goal_is_reached = [0, 0, 0, 0, 0, 0, 0]
        self.state_history = self.initialize_stacked_state_history()

    def initialize_stacked_state_history(self):
        """
        @Lin: 4 stacked frames as RL state
        """
        state_history = np.concatenate((self.obtain_current_state(), self.obtain_current_state()), axis=2)
        state_history = np.concatenate((state_history, self.obtain_current_state()), axis=2)
        state_history = np.concatenate((state_history, self.obtain_current_state()), axis=2)
        return state_history

    def number_of_actions(self):
        return len(self.actions)

    def reset_goal_is_reached(self):
        self.goal_is_reached = [0, 0, 0, 0, 0, 0, 0, 0]

    def start_new_game(self):
        self.gym_env.reset()
        self.life_lost = False
        self.goal_is_reached = [0, 0, 0, 0, 0, 0, 0]
        for i in range(19):
            self.act(0)  # wait for initialization
        self.state_history = self.initialize_stacked_state_history()
        self.agent_previous_X_coord = self.agent_original_location[0]
        self.agent_previous_Y_coord = self.agent_original_location[1]

    def start_new_life(self):
        self.life_lost = False
        self.goal_is_reached = [0, 0, 0, 0, 0, 0, 0]
        for i in range(19):
            self.act(0)  # wait for initialization
        self.state_history = self.initialize_stacked_state_history()
        self.agent_previous_X_coord = self.agent_original_location[0]
        self.agent_previous_Y_coord = self.agent_original_location[1]

    def act(self, action):
        lives = self.ale.lives()
        # print('----',len(self.gym_env.step(self.actions[action])))
        observation, reward, done, truncated, info = self.gym_env.step(self.actions[action])  # done = terminated
        self.life_lost = (not lives == self.ale.lives())
        currState = self.obtain_current_state()
        self.state_history = np.concatenate((self.state_history[:, :, 1:], currState), axis=2)
        return reward

    def getScreen(self):
        screen = self.ale.getScreenGrayscale()
        resized = cv2.resize(screen, (self.screen_width, self.screen_height))
        return resized

    def getScreenRGB(self):
        screen = self.ale.getScreenRGB()
        resized = cv2.resize(screen, (self.screen_width, self.screen_height))
        # resized = screen
        return resized

    def agent_location(self, img):
        #  img = self.getScreenRGB()

        man = [200, 72, 72]
        mask = np.zeros(np.shape(img))
        mask[:, :, 0] = man[0]
        mask[:, :, 1] = man[1]
        mask[:, :, 2] = man[2]

        diff = img - mask
        indxs = np.where(diff == 0)
        diff[np.where(diff < 0)] = 0
        diff[np.where(diff > 0)] = 0
        diff[indxs] = 255
        if (np.shape(indxs[0])[0] == 0):
            mean_x = self.agent_previous_X_coord
            mean_y = self.agent_previous_Y_coord
        else:
            mean_y = np.sum(indxs[0]) / np.shape(indxs[0])[0]
            mean_x = np.sum(indxs[1]) / np.shape(indxs[1])[0]
        self.agent_previous_X_coord = mean_x
        self.agent_previous_Y_coord = mean_y
        return (mean_x, mean_y)

    def devil_location(self, img):
        #    img = self.getScreenRGB()
        # man = [0, 16, 2]
        devilColor = [236, 236, 236]
        mask = np.zeros(np.shape(img))
        mask[:, :, 0] = devilColor[0]
        mask[:, :, 1] = devilColor[1]
        mask[:, :, 2] = devilColor[2]
        diff = img - mask
        indxs = np.where(diff == 0)
        diff[np.where(diff < 0)] = 0
        diff[np.where(diff > 0)] = 0
        diff[indxs] = 255
        if (np.shape(indxs[0])[0] == 0):
            mean_x = self.devil_previous_X_coord
            mean_y = self.devil_previous_Y_coord
        else:
            mean_y = np.sum(indxs[0]) / np.shape(indxs[0])[0]
            mean_x = np.sum(indxs[1]) / np.shape(indxs[1])[0]
        self.devil_previous_X_coord = mean_x
        self.devil_previous_Y_coord = mean_y
        return (mean_x, mean_y)

    def reward_from_distance(self, lastGoal, goal):
        if (lastGoal == -1):
            lastGoalCenter = self.agent_original_location
        else:
            lastGoalCenter = self.goalCenterLoc[lastGoal]
        goalCenter = self.goalCenterLoc[goal]
        agentX, agentY = self.agent_location()
        dis = np.sqrt(
            (goalCenter[0] - agentX) * (goalCenter[0] - agentX) + (goalCenter[1] - agentY) * (goalCenter[1] - agentY))
        disLast = np.sqrt((lastGoalCenter[0] - agentX) * (lastGoalCenter[0] - agentX) + (lastGoalCenter[1] - agentY) * (
                    lastGoalCenter[1] - agentY))
        disGoals = np.sqrt((goalCenter[0] - lastGoalCenter[0]) * (goalCenter[0] - lastGoalCenter[0]) + (
                    goalCenter[1] - lastGoalCenter[1]) * (goalCenter[1] - lastGoalCenter[1]))
        return 0.001 * (disLast - dis) / disGoals

    # add color channel for input of network
    def obtain_current_state(self):
        screen = self.ale.getScreenGrayscale()
        resized = cv2.resize(screen, (self.screen_width, self.screen_height))
        return np.reshape(resized, (84, 84, 1))

    def stack_states_together(self):
        return self.state_history

    def is_game_end(self):
        if self.mode == 'train':
            return self.ale.game_over() or self.life_lost
        return self.ale.game_over()

    def is_game_finished(self):
        return self.ale.game_over()

    def did_agent_die(self):
        return self.life_lost

    def reset(self):
        self.gym_env.reset()
        self.life_lost = False

    def agent_reached_goal(self, goal):
        # if goal in [0,2,4,6]: # those are original task where bounding boxes are used to detect the location of agents
        subset = [0, 2, 3, 4,
                  6]  # those are original task where bounding boxes are used to detect the location of agents
        if goal in subset:
            # goal_index = goal/2
            goal_index = subset.index(goal)
            position_of_goal = self.goalSet[goal_index]
            screen_of_goal = self.initSrcreen
            state_of_screen = self.getScreen()
            count = 0
            for y in range(position_of_goal[0][0], position_of_goal[1][0]):
                for x in range(position_of_goal[0][1], position_of_goal[1][1]):
                    if screen_of_goal[x][y] != state_of_screen[x][y]:
                        count = count + 1
            # 30 is total number of pixels of agent
            if float(count) / 30 > 0.3:
                self.goal_is_reached[goal] = 1
                return True
        if goal == 1:
            # detect if agent is to the left of the devil
            #    return self.devil_on_left()
            return self.find_left_ladder()
        ############## -- DML modified -- ###########
        # if goal == 4:
        #     # detect if agent is to the right of the devil
        # #    return self.devil_on_right()
        #     return self.find_right_ladder()
        ################# -- end -- ###########
        if goal == 5:
            # detect if the agent is back to the original location
            return self.agent_on_original_location()
        return False

    def find_right_ladder(self):
        position_of_goal = self.goalSet[0]
        screen_of_goal = self.initSrcreen
        state_of_screen = self.getScreen()
        count = 0
        for y in range(position_of_goal[0][0], position_of_goal[1][0]):
            for x in range(position_of_goal[0][1], position_of_goal[1][1]):
                if screen_of_goal[x][y] != state_of_screen[x][y]:
                    count = count + 1
        # 30 is total number of pixels of agent
        if float(count) / 30 > 0.3:
            goal = 5
            self.goal_is_reached[goal] = 1
            return True
        return False

    def find_left_ladder(self):
        position_of_goal = self.goalSet[2]
        screen_of_goal = self.initSrcreen
        state_of_screen = self.getScreen()
        count = 0
        for y in range(position_of_goal[0][0], position_of_goal[1][0]):
            for x in range(position_of_goal[0][1], position_of_goal[1][1]):
                if screen_of_goal[x][y] != state_of_screen[x][y]:
                    count = count + 1
        # 30 is total number of pixels of agent
        if float(count) / 30 > 0.3:
            goal = 5
            self.goal_is_reached[goal] = 1
            return True
        return False

    def agent_on_original_location(self):
        img = self.getScreenRGB()
        (x_coord, y_coord) = self.agent_location(img)
        #  print "Agent's location:",x,y
        if abs(x_coord - 42) <= 2 and abs(y_coord - 33) <= 2:
            return True
        else:
            return False

    def pause(self):
        os.system('read -s -n 1 -p "Press any key to continue...\n"')

    def devil_on_left(self):
        img = self.ale.getScreenRGB()
        (x_coord, y_coord) = self.agent_location(img)
        (a_coord, b_coord) = self.devil_location(img)
        #  print "Agent's location:",x,y
        #  print "Devil's location:", a,b
        if (a_coord - x_coord > 40) and (abs(y_coord - b_coord) <= 40):
            return True
        else:
            return False

    def devil_on_right(self):
        img = self.getScreenRGB()
        (x_coord, y_coord) = self.agent_location(img)
        (a_coord, b_coord) = self.devil_location(img)
        # print "Agent's location:",x,y
        # print "Devil's location:",a,b

        # if (x-a > 25) and (abs(y-b) <= 40):
        if (x_coord - a_coord > 40) and (abs(y_coord - b_coord) <= 40):
            return True
        else:
            return False

    def reach_goal_first_time(self, goal):
        if (self.goal_is_reached[goal] == 1):
            return False
        return True


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

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

env = ALEEnvironment(args.game, args)
print("agent loc:",env.agent_location(env.getScreenRGB()))