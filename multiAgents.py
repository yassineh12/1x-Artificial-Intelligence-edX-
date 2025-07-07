# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
       
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        WEIGHT_FOOD = 10.0
        WEIGHT_GHOST = 10.0

        value = successorGameState.getScore()

        distanceToGhost = manhattanDistance(newPos, newGhostStates[0].getPosition())
        if distanceToGhost > 0:
            value -= WEIGHT_GHOST / distanceToGhost

        distancesToFood = [manhattanDistance(newPos, x) for x in newFood.asList()]
        if len(distancesToFood):
            value += WEIGHT_FOOD / min(distancesToFood)

        return value

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        def minimax(agentIndex, depth, state):
            # Terminal condition: game over or max depth
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            numAgents = state.getNumAgents()
            
            # Pacman (Max)
            if agentIndex == 0:
                bestValue = float('-inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = minimax(1, depth, successor)
                    bestValue = max(bestValue, value)
                return bestValue
            else:
                # Ghost (Min)
                nextAgent = (agentIndex + 1) % numAgents
                nextDepth = depth + 1 if nextAgent == 0 else depth
                bestValue = float('inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = minimax(nextAgent, nextDepth, successor)
                    bestValue = min(bestValue, value)
                return bestValue

        bestAction = None
        bestValue = float('-inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = minimax(1, 0, successor)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState):
        def alphabeta(agentIndex, depth, state, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            if agentIndex == 0:
                v = float('-inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    v = max(v, alphabeta(1, depth, successor, alpha, beta))
                    if v > beta:
                        return v
                    alpha = max(alpha, v)
                return v
            else:
                v = float('inf')
                nextAgent = (agentIndex + 1) % numAgents
                nextDepth = depth + 1 if nextAgent == 0 else depth
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    v = min(v, alphabeta(nextAgent, nextDepth, successor, alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
                return v

        bestAction = None
        alpha = float('-inf')
        beta = float('inf')
        v = float('-inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = alphabeta(1, 0, successor, alpha, beta)
            if value > v:
                v = value
                bestAction = action
            alpha = max(alpha, v)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        def expectimax(agentIndex, depth, state):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            if agentIndex == 0:
                v = float('-inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    v = max(v, expectimax(1, depth, successor))
                return v
            else:
                v = 0
                actions = state.getLegalActions(agentIndex)
                prob = 1.0 / len(actions)
                nextAgent = (agentIndex + 1) % numAgents
                nextDepth = depth + 1 if nextAgent == 0 else depth
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    v += prob * expectimax(nextAgent, nextDepth, successor)
                return v

        bestAction = None
        bestValue = float('-inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = expectimax(1, 0, successor)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction


def betterEvaluationFunction(currentGameState):
    """
    A better evaluation function for Pacman.
    """
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghost.scaredTimer for ghost in ghostStates]

    score = currentGameState.getScore()

    if food:
        minFoodDist = min(manhattanDistance(pos, f) for f in food)
        score += 10.0 / (minFoodDist + 1)
    else:
        score += 500

    if capsules:
        minCapDist = min(manhattanDistance(pos, c) for c in capsules)
        score += 5.0 / (minCapDist + 1)

    score -= 4 * len(food)
    score -= 20 * len(capsules)

    for ghost, scared in zip(ghostStates, scaredTimes):
        gPos = ghost.getPosition()
        dist = manhattanDistance(pos, gPos)
        if scared > 0:
            score += 200.0 / (dist + 1)
        else:
            if dist <= 1:
                score -= 500
            else:
                score -= 3.0 / dist

    return score


# Abbreviation
better = betterEvaluationFunction
