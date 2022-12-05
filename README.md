### CSE574 Project: Interpretable and Data-efficient Deep Reinforcement Learning Leveraging Symbolic Planning - https://arxiv.org/pdf/1811.00090.pdf
This project is essentially a reimplementation of the paper. SDRL frameowrk is a way to incorporate symbolic planning so as to aid the hierarchical decision making and focus on meanigful exploration of agent in long horizon sparse reward Atari game Montezuma's Revenge. Moreover, interpretability of the actions taken to achieve the sub tasks is achieved 

# Install Dependencies
# Basic
- python==3.8.0
- tensorflow==2.11.0 or tensorflow-gpu==2.11.0
- gym==0.26.2
- sudo apt-get install gringo
# Extensive 
- pip install -r requirements.txt

# Run
- python train.py
    
## Models 
- https://drive.google.com/drive/folders/1d5GDeR8BCUMiUad0HMvhsWrubb_1RgKM?usp=share_link

## References
- https://arxiv.org/pdf/1604.06057.pdf - Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation
- https://arxiv.org/pdf/1804.08229v1.pdf - An Empirical Comparison of PDDL-based and ASP-based Task Planners
- https://github.com/mgbellemare/Arcade-Learning-Environment#rom-management
- https://github.com/daomingAU/MontezumaRevenge_SDRL
- Deep Reinforcement Learning with Double Q-Learning

