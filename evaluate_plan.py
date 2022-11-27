import sys
import time
import numpy as np
from os import path
import math, random
import planner


def generateplan():
    lppath = "lp/"
    clingopath = "clingo"
    # initial = lppath+"initial.lp"
    goal = lppath+"goal.lp"
    planning = lppath+"montezuma_basic.lp"
    qvalue = lppath+"q.lp"
    constraint = lppath+"constraint.lp"
    return planner.compute_plan(clingopath=clingopath, goal=goal, planning=planning, qvalue=qvalue,
                                constraint=constraint, printout=True)


def calculateplanquality(ro_table, stateaction):
    planquality = 0
    for (state, action) in stateaction:
        planquality += int(math.floor(ro_table[state, action]))
    return planquality


def generate_rovalue_from_table(env, ro_table_lp, ro_table):
    #    print "output qvalues"
    qfile = open("q.lp", "w")
    for (state, action) in ro_table_lp:
        logical_state = stateRemapping(state)
        logical_action = actionRemapping(action)
        qrule = "ro(" + logical_state + "," + logical_action + "," + str(
            int(math.floor(ro_table[state, action]))) + ").\n"
        qfile.write(qrule)
    qfile.close()


def generate_goal_file(planquality):
    #    print "output new goal file"
    goalfile = open("goal.lp", "w")
    goalfile.write("#program check(k).\n")
    #    goalfile.write(":- not at(key,k), query(k).\n")
    goalfile.write(":- query(k), cost(C,k), C <= " + str(planquality) + ".\n")
    goalfile.write(":- query(k), cost(0,k).")
    goalfile.close()


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def cleanupconstraint():
    open('constraint.lp', 'w').close()


def updateconstraint(state_ind, action_ind):
    state = stateRemappingWithTimeStamps(state_ind)
    action = actionRemappingWithTimeStamps(action_ind)
    constraint = ":-" + state + "," + action + ".\n"
    f = open("constraint.lp", "a")
    f.write("#program step(k).\n")
    f.write(constraint)
    f.close()


def selectSubGoal(plantrace, i):
    currentunit = plantrace[i]
    currentfluent = currentunit[2]
    nextunit = plantrace[i + 1]
    nextfluent = nextunit[2]
    # subgoal networks, mapped from every possible symbolic transition
    # currently we only train for good ones. Will add useless(difficult) ones later.
    # make sure the goal number here maps correctly to bounding boxes in environment_atari.py

    if ("at(plat1)" in currentfluent) and ("at(lower_right_ladder)" in nextfluent) and (
            "picked(key)" not in nextfluent):
        return 0
    if ("at(lower_right_ladder)" in currentfluent) and ("at(devilleft)" in nextfluent):
        return 1
    if ("at(devilleft)" in currentfluent) and ("at(key)" in nextfluent):
        return 2
    if ("at(key)" in currentfluent) and ("at(lower_left_ladder)" in nextfluent):
        return 3
    if ("at(lower_left_ladder)" in currentfluent) and ("at(lower_right_ladder)" in nextfluent):
        return 4
    if ("at(lower_right_ladder)" in currentfluent) and ("at(plat1)" in nextfluent):
        return 5
    if ("at(plat1)" in currentfluent) and ("at(right_door)" in nextfluent):
        return 6
    return -1


def obtainStateAction(plantrace, i):
    unit = plantrace[i]
    action = unit[1]
    fluent = unit[2]
    return stateMapping(fluent), actionMapping(action)


def actionMapping(action):
    if 'move(lower_right_ladder)' in action:
        return 0
    if 'move(lower_left_ladder)' in action:
        return 1
    if 'move(key)' in action:
        return 2
    if 'move(right_door)' in action:
        return 3
    if 'move(devilleft)' in action:
        return 4
    if 'move(plat1)' in action:
        return 5


def stateMapping(fluent):  # symbolic state to goal mapping
    if ("at(lower_right_ladder)" in fluent) and ("picked(key)" not in fluent):
        return 0
    if ("at(key)" in fluent) and ("picked(key)" in fluent):
        return 1
    if ("at(lower_right_ladder)" in fluent) and ("picked(key)" in fluent):
        return 2
    if ("at(right_door)" in fluent) and ("picked(key)" in fluent):
        return 3
    if ("at(right_door)" in fluent) and ("picked(key)" not in fluent):
        return 4
    if ("at(devilleft)" in fluent):
        return 5
    if ("at(plat1)" in fluent) and ("picked(key)" in fluent):
        return 6
    if ("at(lower_left_ladder)" in fluent) and ("picked(key)" in fluent):
        return 7
    if ("at(lower_left_ladder)" in fluent) and ("picked(key)" not in fluent):
        return 8
    return -1


def actionRemapping(action_ind):
    if action_ind == 0:
        return 'move(lower_right_ladder)'
    if action_ind == 1:
        return 'move(lower_left_ladder)'
    if action_ind == 2:
        return 'move(key)'
    if action_ind == 3:
        return 'move(right_door)'
    if action_ind == 4:
        return 'move(devilleft)'
    if action_ind == 5:
        return 'move(plat1)'
    return ''


def stateRemapping(fluent_ind):  # symbolic state to goal mapping
    if fluent_ind == -1:
        return 'at(plat1)'
    if fluent_ind == 0:
        return 'at(lower_right_ladder)'
    elif fluent_ind == 1:
        return '(at(key),picked(key))'
    elif fluent_ind == 2:
        return '(at(lower_right_ladder),picked(key))'
    elif fluent_ind == 3:
        return '(at(right_door),picked(key))'
    elif fluent_ind == 4:
        return 'at(right_door)'
    elif fluent_ind == 5:
        return 'at(devilleft)'
    elif fluent_ind == 6:
        return '(at(plat1),picked(key))'
    elif fluent_ind == 7:
        return '(at(lower_left_ladder),picked(key))'
    elif fluent_ind == 8:
        return 'at(lower_left_ladder)'
    return ''


def actionRemappingWithTimeStamps(action_ind):
    if action_ind == 0:
        return 'move(lower_right_ladder,k)'
    if action_ind == 1:
        return 'move(lower_left_ladder,k)'
    if action_ind == 2:
        return 'move(key,k)'
    if action_ind == 3:
        return 'move(right_door,k)'
    if action_ind == 4:
        return 'move(devilleft,k)'
    if action_ind == 5:
        return 'move(plat1,k)'
    return ''


def stateRemappingWithTimeStamps(fluent_ind):  # symbolic state to goal mapping
    if fluent_ind == -1:
        return 'at(plat1,k)'
    if fluent_ind == 0:
        return 'at(lower_right_ladder,k)'
    elif fluent_ind == 1:
        return 'at(key,k),picked(key,k)'
    elif fluent_ind == 2:
        return 'at(lower_right_ladder,k),picked(key,k)'
    elif fluent_ind == 3:
        return 'at(right_door,k),picked(key,k)'
    elif fluent_ind == 4:
        return 'at(right_door,k)'
    elif fluent_ind == 5:
        return 'at(devilleft,k)'
    elif fluent_ind == 6:
        return 'at(plat1,k),picked(key,k)'
    elif fluent_ind == 7:
        return 'at(lower_left_ladder,k),picked(key,k)'
    elif fluent_ind == 8:
        return 'at(lower_left_ladder,k)'
    return ''


def throwdice(threshold):
    rand = random.uniform(0, 1)
    if rand < threshold:
        return True
    else:
        return False


def obtainedKey(previoustate, nextstate):
    if ("picked(key)" not in previoustate) and ("picked(key)" in nextstate):
        return True
    else:
        return False


def openDoor(previoustate, nextstate):
    if ("picked(key)" in previoustate) and ("at(right_door)" not in previoustate) and ("picked(key)" in nextstate) and (
            "at(right_door)" in nextstate):
        return True
    else:
        return False
