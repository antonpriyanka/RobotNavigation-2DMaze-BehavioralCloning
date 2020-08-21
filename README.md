# RobotNavigation-2DMaze-BehavioralCloning

This project aims to demonstrate how classical machine learning methods can be used in robotics setting. In this project, we will working on a navigation agent that navigates inside a simple 2D maze.

![Alt Text](https://github.com/antonpriyanka/RobotNavigation-2DMaze-BehavioralCloning/blob/master/RobotLearningProject1.gif)

The gif above shows the simulation world. The "robot" (also called "agent") is shown by the green dot. The goal location is shown by the red square. The aim of the agent is to navigate to the goal.

The ultimate goal in this project is to learn an appropriate behavior for the agent by imitating demonstrations from an expert user. These demonstrations have been collected controlling the agent via a keyboard.In this project, we will be working on an environment with discrete action space, so we can see behavioral cloning as a classification problem with three output classes (go up, go left, go right). 


The below commands are used to check the scores and visualize the policy learned by the agent.

pipenv run python score_policy.py
pipenv run python score_policy.py --gui
