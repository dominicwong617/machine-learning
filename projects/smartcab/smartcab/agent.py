import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import seaborn as sns
import pdb

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.q_table = dict()
        self.alpha = 0.3
        self.gamma = 0.8
        self.epsilon = 0.1
        self.last_state = None
        self.last_action = None
        self.last_reward = 0

        for light_color in ['red', 'green']:
            for left_traffic in self.env.valid_actions:
                for right_traffic in self.env.valid_actions:
                    for oncoming_traffic in self.env.valid_actions:
                        for next_waypoint in self.env.valid_actions[1:]:
                            state = (light_color, left_traffic, right_traffic, oncoming_traffic, next_waypoint)
                            self.q_table[state] = dict(left=10, right=10, forward=10, none=10)

        self.statistics = dict(success=0, net_reward=0)

    def set_parameters(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def get_q_value(self, state, action):
        if action is None:
            action = 'none'

        return self.q_table[state][action]

    def update_q_value(self, action, reward):
        current_q_value = self.get_q_value(self.last_state, self.last_action)
        self.q_table[self.last_state]['none' if self.last_action is None else self.last_action] = current_q_value + self.alpha * (self.last_reward + self.gamma * self.best_q_value(self.state) - current_q_value)

    def optimal_action(self, state):
        available_actions = self.q_table[state].keys()

        if random.random() > self.epsilon:
            best_q_value = self.best_q_value(self.state)
            available_actions = [action for action in available_actions if self.get_q_value(self.state, action) == best_q_value]

        action = random.choice(available_actions)
        if action == 'none':
            action = None

        return action

    def best_q_value(self, state):
        return max(self.q_table[state].values())

    def get_state_from_inputs(self, inputs, next_waypoint):
        return (inputs['light'], inputs['left'], inputs['right'], inputs['oncoming'], next_waypoint)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.get_state_from_inputs(inputs, self.next_waypoint)

        # TODO: Select action according to your policy
        action = self.optimal_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        if self.last_action is not None and self.last_reward is not None:
            self.update_q_value(action, reward)
        else:
            self.q_table[self.state]['none' if action is None else action] = reward

        self.last_state = self.state
        self.last_action = action
        self.last_reward = reward

        self.statistics['net_reward'] = self.statistics['net_reward'] + reward

        if self.env.trial_data['success'] == 1:
            self.statistics['success'] = self.statistics['success'] + 1

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    parameters_range = np.arange(0.1, 1, 0.1)

    max_success = 0
    max_net_reward = 0
    max_alpha = None
    max_gamma = None
    results = []

    for alpha in parameters_range:
        results.append([])

        for gamma in parameters_range:
            """Run the agent for a finite number of trials."""

            # Set up environment and agent
            e = Environment()  # create environment (also adds some dummy traffic)
            a = e.create_agent(LearningAgent)  # create agent
            a.set_parameters(alpha, gamma)
            e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
            # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

            # Now simulate it
            sim = Simulator(e, update_delay=0.0000001, display=False)  # create simulator (uses pygame when display=True, if available)
            # NOTE: To speed up simulation, reduce update_delay and/or set display=False

            sim.run(n_trials=100)  # run for a specified number of trials
            # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

            print "******************************** Trials statistics ********************************"
            print "Alpha: {}, Gamma: {}".format(alpha, gamma)
            print "Number of successful trials: {}".format(a.statistics['success'])
            print "Net reward: {}".format(a.statistics['net_reward'])
            print "******************************** Trials statistics ********************************"

            results[-1].append(a.statistics['success'])

            if a.statistics['success'] > max_success and a.statistics['net_reward'] > max_net_reward:
                max_success = a.statistics['success']
                max_net_reward = a.statistics['net_reward']
                max_alpha = alpha
                max_gamma = gamma

    print results
    sns.heatmap(results, annot=True, xticklabels=parameters_range, yticklabels=parameters_range)
    sns.plt.savefig("results.png")

    print "================================ Best combination ================================"
    print "Alpha: {}, Gamma: {}".format(max_alpha, max_gamma)
    print "Number of successful trials: {}".format(max_success)
    print "Net reward: {}".format(max_net_reward)
    print "================================ Best combination ================================"


if __name__ == '__main__':
    run()
