# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        # Write value iteration code here
        for n in range(0, iterations): # value iterate n times
            new_values = self.values.copy() # create a copy of the values for this batch
            for s in mdp.getStates(): # iterate through the states
                a = self.computeActionFromValues(s) # get the best action to take from the state
                val = self.computeQValueFromValues(s, a) # get the Q value for the state action pair, using the best action. i.e. gets the new value aagter value iteration
                new_values[s] = val # update
            self.values = new_values # update table after batch

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        if action == None: # if the action is None, return its old value (default=0, as action is always None)
            return self.values[state]
        transitions = self.mdp.getTransitionStatesAndProbs(state, action) # compute the transtitions array [(, prob)]
        q_val = 0
        for (new_state, prob) in transitions:
            q_val += (self.mdp.getReward(state, action, new_state) + self.discount*self.values[new_state])*prob
        return q_val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None
        else:
            best = (None, -696969696) # -inf for default values
            for a in self.mdp.getPossibleActions(state): # iterate through possible actions for the state to find the largest q value
                val = self.computeQValueFromValues(state, a)
                if best[1] <= val: # if greater than other actions
                    best = (a, val) 
            return best[0]


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
