import argparse
import bandits
import eps_greedy
import matplotlib.pyplot as plt
import numpy as np

def argument_handling():
    parser = argparse.ArgumentParser(description="This is a k-armed bandits test with an e-greedy rl method.")
    parser.add_argument("--stationary", "-s", help="set stationary bandits", action="store_true")
    parser.add_argument("--init_estimate", "-i", help="set the initial estimates")
    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument("--k_num", "-k", help="set the number of bandits",required=True)
    requiredNamed.add_argument("--step_size", "-z", help="set the evaluation step size. 'n' for average, number for step size",required=True)
    requiredNamed.add_argument("--steps", "-n", help="set the number of steps",required=True)
    requiredNamed.add_argument("--runs", "-r", help="set the number of runs",required=True)
    requiredNamed.add_argument("--epsilon", "-e", help="set the epsilon for the e-greedy",required=True)

    return parser.parse_args()

def main():
    args = argument_handling()
    stationary = False
    if args.stationary:
        stationary = True
    init_est = 0
    if args.init_estimate:
        init_est = float(args.init_estimate)

    # variables for plot
    steps = range(int(args.steps))
    rewards = [0]*int(args.steps)
    optimal_selection = [0]*int(args.steps)
    
    # start running the test
    for run in range(int(args.runs)):
        # set the k bandits
        k_bandits = bandits.KArmedBandits(int(args.k_num),stationary)
        # set the rl method
        rl_method = eps_greedy.Egreedy(int(args.k_num),init_est,args.step_size,float(args.epsilon))

        for step in range(int(args.steps)):
            action = rl_method.select_action()
            reward = k_bandits.get_bandit_reward(action)
            rewards[step] += reward
            if action == k_bandits.optimal_bandit:
                optimal_selection[step] += 1
            rl_method.update_estimate(action,reward)
    
    rewards = np.array(rewards)/float(args.runs)
    optimal_selection = np.array(optimal_selection)*100/float(args.runs)

    # plot results
    plt.subplot(2, 1, 1)
    plt.plot(steps,rewards)
    plt.ylabel('average reward')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(steps, optimal_selection)
    plt.xlabel('steps')
    plt.ylabel('optimal selection [%]')
    plt.grid()

    plt.show()
    input("Press any key to finish")

if __name__ == "__main__":
    main()