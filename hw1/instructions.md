# CS294-112 HW 1: Imitation Learning

Some additional dependencies (most of which for saving data):
- matplotlib
- seaborn
- pandas
- datetime
- csv

To generate the expert data, one needs to run run_expert.py with arguments pointing to the expert policy and the environment

$ python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2

For the behavioral cloning part of the question, the user needs to run behav_clone.py. It takes in arguments, environment and question part. Valid question aprts are: two, three, three_plot. Three_plot assumes the user had already run part 3

$ python behav_clone.py Humanoid-v2 two

You must generate and save the behavioral cloning data before running the plotting code. Run part two, then part three, then threeplot

For the Dagger question, the user runs DAgger.py similarly. The plot flag generates the plots of Dagger performance.

$ python DAgger.py Hopper-v2 experts/Hopper-v2.pkl --plot
