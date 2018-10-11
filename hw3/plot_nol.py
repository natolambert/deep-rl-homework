import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', nargs='*')
    parser.add_argument('--value', default='AverageReturn', nargs='*')
    args = parser.parse_args()

    for root, dir, files in os.walk(args.logdir[0]):
        for f in files:
            print('Processing file: ', args.logdir[0]+f)

            # if f == '.DS_Store':
            #     pass

            with open(args.logdir[0]+f, 'rb') as d:
                t = []
                mean = []
                max = []
                data = pickle.load(d)
                while True:
                    t.append(data[2])
                    mean.append(data[0])
                    max.append(data[1])
                    try:
                        data = pickle.load(d)
                    except:
                        print('Done reading file')
                        break

                font = {'size'   : 18}

                matplotlib.rc('font', **font)
                matplotlib.rc('lines', linewidth=2.5)

                # plt.tight_layout()

                with sns.axes_style("darkgrid"):
                    ax1 = plt.subplot(111)

                # plt.plot(t,mean, label='Mean Ep Reward, file:' +f )
                plt.plot(t,max, label='Best Mean Ep Reward, file:' +f)
                plt.text(.9*np.max(t), 1.05*np.max(max), 'Best Return: %.2f' % np.max(max))

        plt.title("Learning Curve " + args.logdir[0])
        if args.logdir[0] == 'lunar_dqn_logmore/':
            plt.xlim([0, 500000])
            plt.ylim([-300, 200])
            plt.yticks(np.arange(-300, 201, 50))
            plt.xticks(np.arange(0, 500001, 50000))
        plt.xlabel("Timesteps (t)")
        plt.ylabel("Reward")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
