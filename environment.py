
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
from constants import *
logger = logging.getLogger(__name__)
np.random.seed(SEED)


class ALEEnvironment:
    def __init__(self):
        self.gym_env = gym.make("ALE/MontezumaRevenge-v5",render_mode=RENDER_MODE) 
        self.ale = self.gym_env.ale
        """
        States for DRL agent are stacked 4D tensors in the shape of 4*64*64
        """
        self.histLen = 4

        self.ale.setInt('frame_skip', SKIP_FRAMES)
        self.ale.setFloat('repeat_action_probability', 0.0)
        self.ale.setBool('color_averaging', COLOR_AVERAGING)
        self.ale.setInt('random_seed', 0)  

        if MINIMAL_ACTION_SET:
            self.actions = self.ale.getMinimalActionSet()
            logger.info("Using minimal action set with size %d" % len(self.actions))
        else:
            self.actions = self.ale.getLegalActionSet()
            logger.info("Using full action set with size %d" % len(self.actions))
        logger.debug("Actions: " + str(self.actions))

        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT

        self.mode = "train"
        self.life_lost = False
        self.initSrcreen = self.getScreen()
        print("size of screen is:", self.initSrcreen.shape)
        im = Image.fromarray(self.initSrcreen)

        
        self.goalSet = []
        # goal 0
        self.goalSet.append([[69, 68], [73, 71]])  # Lower Right Ladder.
        # goal 2
        self.goalSet.append([[7, 41], [11, 45]])  # Key. 
        # goal 3
        self.goalSet.append([[11, 68], [15, 71]])  # lower left ladder 
        # goal 4
        self.goalSet.append([[69, 68], [73, 71]])  # Lower Right Ladder again
        # goal 6
        self.goalSet.append([[70, 20], [73, 35]])  # Right Door
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
            self.act(0)  
        self.state_history = self.initialize_stacked_state_history()
        self.agent_previous_X_coord = self.agent_original_location[0]
        self.agent_previous_Y_coord = self.agent_original_location[1]

    def start_new_life(self):
        self.life_lost = False
        self.goal_is_reached = [0, 0, 0, 0, 0, 0, 0]
        for i in range(19):
            self.act(0) 
        self.state_history = self.initialize_stacked_state_history()
        self.agent_previous_X_coord = self.agent_original_location[0]
        self.agent_previous_Y_coord = self.agent_original_location[1]

    def act(self, action):
        lives = self.ale.lives()
        observation, reward, done, truncated, info = self.gym_env.step(self.actions[action]) 
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
       
        return resized

    def agent_location(self, img):
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
        subset = [0, 2, 3, 4, 6] 
        if goal in subset:
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
            return self.find_left_ladder()
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

