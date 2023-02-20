
import math
from collections import Counter
import argparse
import os
import sys
from datetime import datetime
import random
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path
from sumo_rl.exploration.ucb import UCB


def generate_routefile(p1,p2,epi,step):
    d = {}
    random.seed(42)  # make tests reproducible
    N = step  # number of time steps
    # print(N)
    # demand per second from different directions
    veh_n_0 = 0
    veh_n_1 = 0
    veh_w_0 = 0
    veh_w_1 = 0
    #speedFactor="1" speedDev="0" departSpeed="speedLimit"
    # The driver imperfection (0 denotes perfect driving)
    # print("road_type:", road_type)
    
    path = Path("nets/single-intersection/two_way/single-intersection_test{}.rou.xml".format(epi)).absolute()
    with path.open("w") as routes:
        print("""<routes>
        <vType id="typeWE" accel="2.6" decel="4.5" sigma="0" length="5" minGap="2.5" maxSpeed="15.65" \
guiShape="passenger"/>
        <vType id="typeEW" accel="2.6" decel="4.5" sigma="0" length="5" minGap="2.5" maxSpeed="15.65" \
guiShape="passenger"/>
        <vType id="typeNS" accel="2.6" decel="4.5" sigma="0" length="5" minGap="2.5" maxSpeed="15.65" \
        guiShape="passenger"/>
        <vType id="typeSN" accel="2.6" decel="4.5" sigma="0" length="5" minGap="2.5" maxSpeed="15.65" \
        guiShape="passenger"/>
        <route id="eastbound" edges="1i 2i" />
        <route id="westbound" edges="-2i -1i" />
        <route id="northbound" edges="-3i 4i" />
        <route id="southbound" edges="-4i 3i" />
        """, file=routes)
        vehNr = 0
        for i in range(N):
            if random.uniform(0, 1) < p2:
                print('    <vehicle id="eastbound_%i" type="typeWE" route="eastbound" depart="%i" departSpeed="max" />' % (
                    vehNr, i), file=routes)
                d['eastbound_{}'.format(vehNr)] = i
                veh_w_0 += 1
                vehNr += 1
            if random.uniform(0, 1) < p2:
                print('    <vehicle id="westbound_%i" type="typeEW" route="westbound" depart="%i" departSpeed="max"/>' % (
                    vehNr, i), file=routes)
                d['westbound_{}'.format(vehNr)] = i
                veh_w_1 += 1
                vehNr += 1
            if random.uniform(0, 1) < p1:
                print('    <vehicle id="southbound_%i" type="typeNS" route="southbound" depart="%i" departSpeed="max"/>' % (
                    vehNr, i), file=routes)
                d['southbound_{}'.format(vehNr)] = i
                veh_n_0 += 1
                vehNr += 1
            if random.uniform(0, 1) < p1:
                print('    <vehicle id="northbound_%i" type="typeSN" route="northbound" depart="%i" departSpeed="max"/>' % (
                    vehNr, i), file=routes)
                d['northbound_{}'.format(vehNr)] = i
                veh_n_1 += 1
                vehNr += 1
        print("</routes>", file=routes)

    return d, vehNr

if __name__ == '__main__':
    for i in range(2):
        d, vehNr = generate_routefile(500/3600,900/3600,i,14400)
       
