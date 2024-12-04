# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random

from setuptools.command.bdist_egg import can_scan

import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='AgentRamenDon', second='AgentYippyYap', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########


class SecretBaseAgent(CaptureAgent):

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        # define patrol points
        self.x, self.y = self.get_patrol_points(game_state)
        self.improve_patrol_points(game_state)
        self.starting_pos = game_state.get_agent_position(self.index)

    # OVERRIDE: each agent defines their own patrol points
    def get_patrol_points(self, game_state):
        raise NotImplementedError("implement this method!!!")

    def improve_patrol_points(self, game_state):
        height = game_state.data.layout.height
        width = game_state.data.layout.width


        # normalize coordinates to prevent errors
        if self.x < 0:
            self.x += width
        if self.y < 0:
            self.y += height
        self.x = self.x % width
        self.y = self.y % height

        walls = game_state.get_walls()
        # if patrol point is wall, change y-coordinate
        while walls[self.x][self.y]:
            self.y = (self.y + 1) % height


    def choose_action(self, game_state):
        pos = game_state.get_agent_position(self.index)
        actions = game_state.get_legal_actions(self.index)

        # opponents + their states
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        # if invaders => chase them
        if invaders:
            closest_invader = self.get_closest_invader(pos, invaders)
            return self.get_to_target(game_state, closest_invader, actions)

        # otherwise border patrol
        return self.get_to_target(game_state, (self.x, self.y), actions)

    def get_to_target(self, game_state, target, actions):
        best_action = None
        min_distance = float('inf')
        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            successor_position = successor.get_agent_position(self.index)
            distance = self.get_maze_distance(successor_position, target)
            if distance < min_distance:
                min_distance = distance
                best_action = action
        return best_action

    # assume invaders != empty
    def get_closest_invader(self, pos, invaders):
        closest_invader = None
        min_distance = float('inf')

        # find closest invader out of all invaders
        for invader in invaders:
            invader_position = invader.get_position()
            distance = self.get_maze_distance(pos, invader_position)
            if distance < min_distance:
                min_distance = distance
                closest_invader = invader_position
        return closest_invader


# complete defensive agent, border patrol in the middle
class AgentYippyYap(SecretBaseAgent):

    # set custom border patrol point
    def get_patrol_points(self, game_state):
        layout = game_state.data.layout
        x = layout.width // 2
        y = layout.height // 2

        # adjust coordinates depending on color
        if self.red:
            x -= 1
        else:
            x += 0
            y -= 1

        return x, y




class AgentRamenDon(SecretBaseAgent):

    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.mode = "attack"  # Start in attack mode
        self.power_mode_active = False  # track if power capsule is active
        self.successful_steal = False  # track successful stealing attempts
        self.safety_count= 1

    def choose_action(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        stolen_food = game_state.get_agent_state(self.index).num_carrying
        legal_actions = game_state.get_legal_actions(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        power_capsule_active = any(enemy.scared_timer > 0 for enemy in enemies)
        ghosts = [enemy for enemy in enemies if not enemy.is_pacman and enemy.get_position()]

        # if agent respawns set to attack mode
        if game_state.get_agent_position(self.index) == self.starting_pos:
            self.mode = "attack"


        # case of power capsule activation
        if power_capsule_active:
            self.power_mode_active = True
            self.mode = "attack"  # full attack mode during activation
        elif self.power_mode_active and not power_capsule_active:
            self.power_mode_active = False  # Reset power mode after end of usage
            self.mode = "defense" if stolen_food == 0 and self.successful_steal else "attack"

        # adjust pacman path based on ghost proximity
        self.safety_count = self.update_safety_count(my_pos, ghosts)

        # choose between attack and defense
        if self.mode == "attack":
            return self.attack_behavior(game_state, my_pos, legal_actions, stolen_food, ghosts)
        else:
        # defense inherited from SecretBaseAgent
            return super().choose_action(game_state)

    def attack_behavior(self, game_state, my_pos, legal_actions, stolen_food, ghosts):

        food_list = self.get_food(game_state).as_list()

        # go back to border point if carrying enough food or in danger
        if stolen_food >= 2 or self.is_in_danger(my_pos, ghosts, game_state):
            # track successful stealing if carrying food
            if stolen_food > 0:
                self.successful_steal = True
                # switch from attack to defense after successfull steal
                self.mode = "defense"
            return self.to_border_point(game_state, legal_actions)

        # collect safest food to prevent respawn
        target_food = self.select_safe_food(my_pos, food_list, ghosts)
        if target_food:
            return self.get_to_target(game_state, target_food, legal_actions)

        # random move if actions available, stop if no legal actions existing
        return random.choice(legal_actions) if legal_actions else Directions.STOP

    def update_safety_count(self, my_pos, ghosts):

        distance_ghost = [
            self.get_maze_distance(my_pos, ghost.get_position())
            for ghost in ghosts if ghost.get_position()
        ]
        if not distance_ghost:
            return 1  # low safety count when no ghost nearby
        closest_distance = min(distance_ghost)

        base_safety = 2
        scale_factor = 3  # adjusts dynamic increase
        safety_count = base_safety + int(scale_factor * (1 / (closest_distance + 1)) ** 2) # exponential switch
        return min(safety_count, 10)  # max safety count of 10

    def is_in_danger(self, my_pos, ghosts, game_state):

       # evaluate danger state through distance
        for ghost in ghosts:
            ghost_pos = ghost.get_position()
            if ghost_pos and self.get_maze_distance(my_pos, ghost_pos) < 4:
                return True
        return False

    def select_safe_food(self, my_pos, food_list, ghosts):

        score_food = []
        for food in food_list:
            score = 20
            distance = self.get_maze_distance(my_pos, food)
            score -= distance  # high penalty for further food to ensure short path

            for ghost in ghosts:
                ghost_pos = ghost.get_position()
                if ghost_pos and self.get_maze_distance(food, ghost_pos) < self.safety_count:
                    score -= 30  # penalty for closer ghosts

            score_food.append((score, food))

        # randomly choose highest food to make Ramen don unpredictable
        score_food.sort(reverse=True, key=lambda x: x[0])
        top_food_options = score_food[:3]
        if top_food_options:
            return random.choice(top_food_options)[1]  # pick one out of three best
        else:
            return None

    def to_border_point(self, game_state, legal_actions):
        return self.get_to_target(game_state, (self.x, self.y), legal_actions)

    def get_patrol_points(self, game_state):
        layout = game_state.data.layout
        x = layout.width // 2
        y = layout.height // 2

        # adjust coordinates depending on color
        if self.red:
            x -= 1
            y -= 4
        else:
            x += 0
            y += 3

        return x, y

