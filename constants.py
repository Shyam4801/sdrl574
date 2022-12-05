actionMap = [0, 1, 2, 3, 4, 5, 11, 12]

actionExplain = ['no action', 'jump', 'up', 'right', 'left', 'down', 'jump right', 'jump left']

goalExplain = ['lower right ladder', 'jump to the left of devil', 'key', 'lower left ladder','lower right ladder', 'central high platform', 'right door']  


GAME = "montezuma_revenge.bin"
DISPLAY = True
SKIP_FRAMES = 4
COLOR_AVERAGING = True
RANDOM_SEED = 0
MINIMAL_ACTION_SET = False
SCREEN_WIDTH=84
SCREEN_HEIGHT=84
LOAD_WEIGHT = False
USE_SPARSE_REWARD  = True
RENDER_MODE = "rgb_array"

# Default parameters for the Agent
defaultEpsilon = 1.0
defaultControllerEpsilon = 1.0

maxReward = 1
minReward = -1

prioritized_replay_alpha = 0.6
max_timesteps=1000000
prioritized_replay_beta0=0.4
prioritized_replay_eps=1e-6
prioritized_replay_beta_iters = max_timesteps*0.5

# Default parameters for the controllerNet
nb_Action = 8