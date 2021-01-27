from __future__ import print_function

import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw
import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import scipy

st.sidebar.button("Re Run")
ghost = st.sidebar.checkbox("Are you a ghost?", value=False, key=None)


#Anaconda
#cd /D C:\Users\Abraham\miniconda3\envs\snowflakes\Scripts
#streamlit run amongus.py

def create_data_model(distance_matrix, starting):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = distance_matrix  # yapf: disable
    data['distance_matrix'] = [[int(100*i) for i in m] for m in data['distance_matrix']]
    data['num_vehicles'] = 1
    data['starts'] = [starting]
    data['ends'] = [len(data['distance_matrix'])-1]
    return data


def drop_false(truths):
    global ghost
    placematrix = np.array(['Admin', 'Cafeteria', 'Communcations', 'Electrical', 'Lower engines', 'Medbay', 'Navigation', 'O2',
                 'Reactor', 'Security', 'Shields', 'Storage', 'Upper engines', 'Weapons'])
    pointmatrix = np.array(['(1550, 730)', '(1297, 317)', '(1580, 1200)', '(900, 900)', '(400, 1000)', '(850, 500)', '(2200, 600)', '(1650, 550)', '(200, 650)', '(650, 650)', '(1800, 1000)', '(1230, 1060)', '(420, 320)', '(1800, 300)'])
    new_bad_names = {}
    for i in range(len(placematrix)):
        new_bad_names[placematrix[i]] = pointmatrix[i]
    if not ghost:
        dmatrix = np.array([
            [0, 5.07, 7.57, 9.74, 11.35, 9.5, 13.01, 11.17, 15.16, 14.69, 7.62, 3.94, 10.75, 8.42, 0],
            [5.07, 0, 8.32, 10.49, 12.18, 5.25, 9.33, 7.1, 10.84, 10.52, 8.68, 5.1, 6.15, 4.23, 0],
            [7.57, 8.32, 0, 9.71, 11.14, 12.66, 8.21, 9.01, 15.83, 15.69, 2.64, 3.82, 13.92, 9.82, 0],
            [9.74, 10.49, 9.71, 0, 5.26, 15.45, 16.27, 16.67, 10.96, 9.8, 10.58, 4.96, 11.39, 14.19, 0],
            [11.35, 12.18, 11.14, 5.26, 0, 9.95, 17.38, 18.7, 4.78, 4.35, 12.15, 7.07, 5.86, 16.22, 0],
            [9.5, 5.25, 12.66, 15.45, 9.95, 0, 14.51, 12.17, 8.85, 8.15, 13.18, 9.38, 4.42, 9.28, 0],
            [13.01, 9.33, 8.21, 16.27, 17.38, 14.51, 0, 4.47, 20.52, 19.75, 5.62, 9.53, 15.73, 5.17, 0],
            [11.17, 7.1, 9.01, 16.67, 18.7, 12.17, 4.47, 0, 18.3, 17.62, 6.52, 10.32, 13.67, 3.01, 0],
            [15.16, 10.84, 15.83, 10.96, 4.78, 8.85, 20.52, 18.3, 0, 3.34, 16.5, 10.8, 4.62, 11.01, 0],
            [14.69, 10.52, 15.69, 9.8, 4.35, 8.15, 19.75, 17.62, 3.34, 0, 16.27, 10.57, 4.31, 10.72, 0],
            [7.62, 8.68, 2.64, 10.58, 12.15, 13.18, 5.62, 6.52, 16.5, 16.27, 0, 4.16, 14.31, 7.15, 0],
            [3.94, 5.1, 3.82, 4.96, 7.07, 9.38, 9.53, 10.32, 10.8, 10.57, 4.16, 0, 10.46, 8.11, 0],
            [10.75, 6.15, 13.92, 11.39, 5.86, 4.42, 15.73, 13.67, 4.62, 4.31, 14.31, 10.46, 0, 10.15, 0],
            [8.42, 4.23, 9.82, 14.19, 16.22, 9.28, 5.17, 3.01, 11.01, 10.72, 7.15, 8.11, 10.15, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
    else:
        dmatrix = np.array([
            [0, 545.25, 401.123, 682.367, 1167.262, 761.577, 680.074, 269.258, 1358.308, 912.414, 320.156, 412.311, 1227.721, 559.017, 0],
            [545.25, 0, 927.242, 719.703, 1127.43, 483.009, 946.308, 422.963, 1146.428, 727.666, 848.232, 746.015, 877.005, 503.287, 0],
            [401.123, 927.242, 0, 766.176, 1196.829, 1011.385, 862.786, 653.758, 1485.564, 1080.463, 297.321, 376.962, 1456.022, 926.499, 0],
            [682.367, 719.703, 766.176, 0, 485.412, 400.78, 1358.538, 850.368, 719.809, 336.341, 930.39, 389.391, 737.174, 1102.554, 0],
            [1167.262, 1127.43, 1196.829, 485.412, 0, 672.681, 1843.909, 1328.533, 403.113, 430.116, 1400, 832.166, 680.294, 1565.248, 0],
            [761.577, 483.009, 1011.385, 400.78, 672.681, 0, 1353.699, 801.561, 667.083, 250, 1073.546, 676.757, 466.154, 970.824, 0],
            [680.074, 946.308, 862.786, 1358.538, 1843.909, 1353.699, 0, 552.268, 2000.625, 1550.806, 565.685, 1073.546, 1801.888, 500, 0],
            [269.258, 422.963, 653.758, 850.368, 1328.533, 801.561, 552.268, 0, 1453.444, 1004.988, 474.342, 660.681, 1251.319, 291.548, 0],
            [1358.308, 1146.428, 1485.564, 719.809, 403.113, 667.083, 2000.625, 1453.444, 0, 450, 1637.834, 1108.603, 396.611, 1637.834, 0],
            [912.414, 727.666, 1080.463, 336.341, 430.116, 250, 1550.806, 1004.988, 450, 0, 1202.082, 710.282, 402.244, 1202.082, 0],
            [320.156, 848.232, 297.321, 930.39, 1400, 1073.546, 565.685, 474.342, 1637.834, 1202.082, 0, 573.149, 1538.441, 700, 0],
            [412.311, 746.015, 376.962, 389.391, 832.166, 676.757, 1073.546, 660.681, 1108.603, 710.282, 573.149, 0, 1097.133, 950, 0],
            [1227.721, 877.005, 1456.022, 737.174, 680.294, 466.154, 1801.888, 1251.319, 396.611, 402.244, 1538.441, 1097.133, 0, 1380.145, 0],
            [559.017, 503.287, 926.499, 1102.554, 1565.248, 970.824, 500, 291.548, 1637.834, 1202.082, 700, 950, 1380.145, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype = object)
    false_list = list(filter(lambda i: not truths[i], range(len(truths))))
    placematrix = np.delete(placematrix, false_list, 0)
    #st.write("You have tasks in: \n")
    #pointmatrix = np.delete(pointmatrix, false_list, 0)
    dmatrix = np.delete(dmatrix, false_list, 0)
    dmatrix = np.delete(dmatrix, false_list, 1)
    res = dict((k, new_bad_names[k]) for k in placematrix
               if k in new_bad_names)
    return list(dmatrix), list(placematrix), pointmatrix, res, ghost

def return_pathing_tuples(start, stop):
    pathing_matrix = [[[0], [(1297, 317), (1280, 728), (1550, 800)], [(1580, 1200), (1591, 1000), (1230, 1060), (1280, 728), (1550, 800)], [(1550, 800), (1280, 728), (1230, 1060), (820, 1156), (875, 900)], [(1550, 800), (1280, 728), (1230, 1060), (820, 1156), (650, 1013), (400, 1000)], [(1550, 800), (1280, 728), (1297, 317), (840, 300), (850, 500)], [(1550, 800), (1280, 728), (1297, 317), (1800, 300), (1821, 540), (2011, 600), (2200, 600)], [(1550, 800), (1280, 728), (1297, 317), (1800, 300), (1821, 540), (1650, 550)], [(1550, 800), (1280, 728), (1297, 317), (840, 300), (420, 320), (430, 630), (200, 650)], [(1550, 800), (1280, 728), (1297, 317), (840, 300), (420, 320), (430, 630), (650, 650)], [(1550, 800), (1280, 728), (1230, 1060), (1591, 1000), (1800, 1000)], [(1550, 800), (1280, 728), (1230, 1060)], [(1550, 800), (1280, 728), (1297, 317), (840, 300), (420, 320)], [(1550, 800), (1280, 728), (1297, 317), (1800, 300)]],
[[(1297, 317), (1280, 728), (1550, 800)], [0], [(1297, 317), (1280, 728), (1230, 1060), (1591, 1000), (1580, 1200)], [(1297, 317), (1280, 728), (1230, 1060), (820, 1156), (875, 900)], [(1297, 317), (840, 300), (420, 320), (430, 630), (400, 1000)], [(1297, 317), (840, 300), (850, 500)], [(1297, 317), (1800, 300), (1821, 540), (2011, 600), (2200, 600)], [(1297, 317), (1800, 300), (1821, 540), (1650, 550)], [(1297, 317), (840, 300), (420, 320), (430, 630), (200, 650)], [(1297, 317), (840, 300), (420, 320), (430, 630), (650, 650)], [(1297, 317), (1280, 728), (1230, 1060), (1591, 1000), (1800, 1000)], [(1297, 317), (1280, 728), (1230, 1060)], [(1297, 317), (840, 300), (420, 320)], [(1297, 317), (1800, 300)]],
[[(1580, 1200), (1591, 1000), (1230, 1060), (1280, 728), (1550, 800)], [(1297, 317), (1280, 728), (1230, 1060), (1591, 1000), (1580, 1200)], [0], [(1580, 1200), (1591, 1000), (1230, 1060), (820, 1156), (875, 900)], [(1580, 1200), (1591, 1000), (1230, 1060), (820, 1156), (650, 1013), (400, 1000)], [(1580, 1200), (1591, 1000), (1230, 1060), (1280, 728), (1297, 317), (840, 300), (850, 500)], [(1580, 1200), (1591, 1000), (1800, 1000), (1820, 700), (2011, 600), (2200, 600)], [(1580, 1200), (1591, 1000), (1800, 1000), (1820, 700), (2011, 600), (1821, 540), (1650, 550)], [(1580, 1200), (1591, 1000), (1230, 1060), (820, 1156), (650, 1013), (400, 1000), (430, 630), (200, 650)], [(1580, 1200), (1591, 1000), (1230, 1060), (820, 1156), (650, 1013), (400, 1000), (430, 630), (650, 650)], [(1580, 1200), (1591, 1000), (1800, 1000)], [(1580, 1200), (1591, 1000), (1230, 1060)], [(1580, 1200), (1591, 1000), (1230, 1060), (1280, 728), (1297, 317), (840, 300), (420, 320)], [(1580, 1200), (1591, 1000), (1800, 1000), (1820, 700), (2011, 600), (1821, 540), (1800, 300)]],
[[(1550, 800), (1280, 728), (1230, 1060), (820, 1156), (875, 900)], [(1297, 317), (1280, 728), (1230, 1060), (820, 1156), (875, 900)], [(1580, 1200), (1591, 1000), (1230, 1060), (820, 1156), (875, 900)], [0], [(875, 900), (820, 1156), (650, 1013), (400, 1000)], [(875, 900), (820, 1156), (650, 1013), (400, 1000), (430, 630), (420, 320), (840, 300), (850, 500)], [(875, 900), (820, 1156), (1230, 1060), (1591, 1000), (1800, 1000), (1820, 700), (2011, 600), (2200, 600)], [(875, 900), (820, 1156), (1230, 1060), (1591, 1000), (1800, 1000), (1820, 700), (2011, 600), (1821, 540), (1650, 550)], [(875, 900), (820, 1156), (650, 1013), (400, 1000), (430, 630), (200, 650)], [(875, 900), (820, 1156), (650, 1013), (400, 1000), (430, 630), (650, 650)], [(875, 900), (820, 1156), (1230, 1060), (1591, 1000), (1800, 1000)], [(875, 900), (820, 1156), (1230, 1060)], [(875, 900), (820, 1156), (650, 1013), (400, 1000), (430, 630), (420, 320)], [(875, 900), (820, 1156), (1230, 1060), (1280, 728), (1297, 317), (1800, 300)]],
[[(1550, 800), (1280, 728), (1230, 1060), (820, 1156), (650, 1013), (400, 1000)], [(1297, 317), (840, 300), (420, 320), (430, 630), (400, 1000)], [(1580, 1200), (1591, 1000), (1230, 1060), (820, 1156), (650, 1013), (400, 1000)], [(875, 900), (820, 1156), (650, 1013), (400, 1000)], [0], [(400, 1000), (430, 630), (420, 320), (840, 300), (850, 500)], [(400, 1000), (650, 1013), (820, 1156), (1230, 1060), (1591, 1000), (1800, 1000), (1820, 700), (2011, 600), (2200, 600)], [(400, 1000), (650, 1013), (820, 1156), (1230, 1060), (1591, 1000), (1800, 1000), (1820, 700), (2011, 600), (1821, 540), (1650, 550)], [(400, 1000), (430, 630), (200, 650)], [(400, 1000), (430, 630), (650, 650)], [(400, 1000), (650, 1013), (820, 1156), (1230, 1060), (1591, 1000), (1800, 1000)], [(400, 1000), (650, 1013), (820, 1156), (1230, 1060)], [(400, 1000), (430, 630), (420, 320)], [(400, 1000), (430, 630), (420, 320), (840, 300), (1297, 317), (1800, 300)]],
[[(1550, 800), (1280, 728), (1297, 317), (840, 300), (850, 500)], [(1297, 317), (840, 300), (850, 500)], [(1580, 1200), (1591, 1000), (1230, 1060), (1280, 728), (1297, 317), (840, 300), (850, 500)], [(875, 900), (820, 1156), (650, 1013), (400, 1000), (430, 630), (420, 320), (840, 300), (850, 500)], [(400, 1000), (430, 630), (420, 320), (840, 300), (850, 500)], [0], [(850, 500), (840, 300), (1297, 317), (1800, 300), (1821, 540), (2011, 600), (2200, 600)], [(850, 500), (840, 300), (1297, 317), (1800, 300), (1821, 540), (1650, 550)], [(850, 500), (840, 300), (420, 320), (430, 630), (200, 650)], [(850, 500), (840, 300), (420, 320), (430, 630), (650, 650)], [(850, 500), (840, 300), (1297, 317), (1280, 728), (1230, 1060), (1591, 1000), (1800, 1000)], [(850, 500), (840, 300), (1297, 317), (1280, 728), (1230, 1060)], [(850, 500), (840, 300), (420, 320)], [(850, 500), (840, 300), (1297, 317), (1800, 300)]],
[[(1550, 800), (1280, 728), (1297, 317), (1800, 300), (1821, 540), (2011, 600), (2200, 600)], [(1297, 317), (1800, 300), (1821, 540), (2011, 600), (2200, 600)], [(1580, 1200), (1591, 1000), (1800, 1000), (1820, 700), (2011, 600), (2200, 600)], [(875, 900), (820, 1156), (1230, 1060), (1591, 1000), (1800, 1000), (1820, 700), (2011, 600), (2200, 600)], [(400, 1000), (650, 1013), (820, 1156), (1230, 1060), (1591, 1000), (1800, 1000), (1820, 700), (2011, 600), (2200, 600)], [(850, 500), (840, 300), (1297, 317), (1800, 300), (1821, 540), (2011, 600), (2200, 600)], [0], [(2200, 600), (2011, 600), (1821, 540), (1650, 550)], [(2200, 600), (2011, 600), (1821, 540), (1800, 300), (1297, 317), (840, 300), (420, 320), (430, 630), (200, 650)], [(2200, 600), (2011, 600), (1821, 540), (1800, 300), (1297, 317), (840, 300), (420, 320), (430, 630), (650, 650)], [(2200, 600), (2011, 600), (1820, 700), (1800, 1000)], [(2200, 600), (2011, 600), (1820, 700), (1800, 1000), (1591, 1000), (1230, 1060)], [(2200, 600), (2011, 600), (1821, 540), (1800, 300)], [(2200, 600), (2011, 600), (1821, 540), (1800, 300)]],
[[(1550, 800), (1280, 728), (1297, 317), (1800, 300), (1821, 540), (1650, 550)], [(1297, 317), (1800, 300), (1821, 540), (1650, 550)], [(1580, 1200), (1591, 1000), (1800, 1000), (1820, 700), (2011, 600), (1821, 540), (1650, 550)], [(875, 900), (820, 1156), (1230, 1060), (1591, 1000), (1800, 1000), (1820, 700), (2011, 600), (1821, 540), (1650, 550)], [(400, 1000), (650, 1013), (820, 1156), (1230, 1060), (1591, 1000), (1800, 1000), (1820, 700), (2011, 600), (1821, 540), (1650, 550)], [(850, 500), (840, 300), (1297, 317), (1800, 300), (1821, 540), (1650, 550)], [(2200, 600), (2011, 600), (1821, 540), (1650, 550)], [0], [(1650, 550), (1821, 540), (1800, 300), (1297, 317), (840, 300), (420, 320), (430, 630), (200, 650)], [(1650, 550), (1821, 540), (1800, 300), (1297, 317), (840, 300), (420, 320), (430, 630), (650, 650)], [(1650, 550), (1821, 540), (2011, 600), (1820, 700), (1800, 1000)], [(1650, 550), (1821, 540), (2011, 600), (1820, 700), (1800, 1000), (1591, 1000), (1230, 1060)], [(1650, 550), (1821, 540), (1800, 300), (1297, 317), (840, 300), (420, 320)], [(1650, 550), (1821, 540), (1800, 300)]],
[[(1550, 800), (1280, 728), (1297, 317), (840, 300), (420, 320), (430, 630), (200, 650)], [(1297, 317), (840, 300), (420, 320), (430, 630), (200, 650)], [(1580, 1200), (1591, 1000), (1230, 1060), (820, 1156), (650, 1013), (400, 1000), (430, 630), (200, 650)], [(875, 900), (820, 1156), (650, 1013), (400, 1000), (430, 630), (200, 650)], [(400, 1000), (430, 630), (200, 650)], [(850, 500), (840, 300), (420, 320), (430, 630), (200, 650)], [(2200, 600), (2011, 600), (1821, 540), (1800, 300), (1297, 317), (840, 300), (420, 320), (430, 630), (200, 650)], [(1650, 550), (1821, 540), (1800, 300), (1297, 317), (840, 300), (420, 320), (430, 630), (200, 650)], [0], [(200, 650), (430, 630), (650, 650)], [(200, 650), (430, 630), (400, 1000), (650, 1013), (820, 1156), (1230, 1060), (1591, 1000), (1800, 1000)], [(200, 650), (430, 630), (400, 1000), (650, 1013), (820, 1156), (1230, 1060)], [(200, 650), (430, 630), (420, 320)], [(200, 650), (430, 630), (420, 320), (840, 300), (1297, 317), (1800, 300)]],
[[(1550, 800), (1280, 728), (1297, 317), (840, 300), (420, 320), (430, 630), (650, 650)], [(1297, 317), (840, 300), (420, 320), (430, 630), (650, 650)], [(1580, 1200), (1591, 1000), (1230, 1060), (820, 1156), (650, 1013), (400, 1000), (430, 630), (650, 650)], [(875, 900), (820, 1156), (650, 1013), (400, 1000), (430, 630), (650, 650)], [(400, 1000), (430, 630), (650, 650)], [(850, 500), (840, 300), (420, 320), (430, 630), (650, 650)], [(2200, 600), (2011, 600), (1821, 540), (1800, 300), (1297, 317), (840, 300), (420, 320), (430, 630), (650, 650)], [(1650, 550), (1821, 540), (1800, 300), (1297, 317), (840, 300), (420, 320), (430, 630), (650, 650)], [(200, 650), (430, 630), (650, 650)], [0], [(650, 650), (430, 630), (400, 1000), (650, 1013), (820, 1156), (1230, 1060), (1591, 1000), (1800, 1000)], [(650, 650), (430, 630), (400, 1000), (650, 1013), (820, 1156), (1230, 1060)], [(650, 650), (430, 630), (420, 320)], [(650, 650), (430, 630), (420, 320), (840, 300), (1297, 317), (1800, 300)]],
[[(1550, 800), (1280, 728), (1230, 1060), (1591, 1000), (1800, 1000)], [(1297, 317), (1280, 728), (1230, 1060), (1591, 1000), (1800, 1000)], [(1580, 1200), (1591, 1000), (1800, 1000)], [(875, 900), (820, 1156), (1230, 1060), (1591, 1000), (1800, 1000)], [(400, 1000), (650, 1013), (820, 1156), (1230, 1060), (1591, 1000), (1800, 1000)], [(850, 500), (840, 300), (1297, 317), (1280, 728), (1230, 1060), (1591, 1000), (1800, 1000)], [(2200, 600), (2011, 600), (1820, 700), (1800, 1000)], [(1650, 550), (1821, 540), (2011, 600), (1820, 700), (1800, 1000)], [(200, 650), (430, 630), (400, 1000), (650, 1013), (820, 1156), (1230, 1060), (1591, 1000), (1800, 1000)], [(650, 650), (430, 630), (400, 1000), (650, 1013), (820, 1156), (1230, 1060), (1591, 1000), (1800, 1000)], [0], [(1800, 1000), (1591, 1000), (1230, 1060)], [(1800, 1000), (1591, 1000), (1230, 1060), (1280, 728), (1297, 317), (840, 300), (420, 320)], [(1800, 1000), (1820, 700), (2011, 600), (1821, 540), (1800, 300)]],
[[(1550, 800), (1280, 728), (1230, 1060)], [(1297, 317), (1280, 728), (1230, 1060)], [(1580, 1200), (1591, 1000), (1230, 1060)], [(875, 900), (820, 1156), (1230, 1060)], [(400, 1000), (650, 1013), (820, 1156), (1230, 1060)], [(850, 500), (840, 300), (1297, 317), (1280, 728), (1230, 1060)], [(2200, 600), (2011, 600), (1820, 700), (1800, 1000), (1591, 1000), (1230, 1060)], [(1650, 550), (1821, 540), (2011, 600), (1820, 700), (1800, 1000), (1591, 1000), (1230, 1060)], [(200, 650), (430, 630), (400, 1000), (650, 1013), (820, 1156), (1230, 1060)], [(650, 650), (430, 630), (400, 1000), (650, 1013), (820, 1156), (1230, 1060)], [(1800, 1000), (1591, 1000), (1230, 1060)], [0], [(1230, 1060), (1280, 728), (1297, 317), (840, 300), (420, 320)], [(1230, 1060), (1280, 728), (1297, 317), (1800, 300)]],
[[(1550, 800), (1280, 728), (1297, 317), (840, 300), (420, 320)], [(1297, 317), (840, 300), (420, 320)], [(1580, 1200), (1591, 1000), (1230, 1060), (1280, 728), (1297, 317), (840, 300), (420, 320)], [(875, 900), (820, 1156), (650, 1013), (400, 1000), (430, 630), (420, 320)], [(400, 1000), (430, 630), (420, 320)], [(850, 500), (840, 300), (420, 320)], [(2200, 600), (2011, 600), (1821, 540), (1800, 300)], [(1650, 550), (1821, 540), (1800, 300), (1297, 317), (840, 300), (420, 320)], [(200, 650), (430, 630), (420, 320)], [(650, 650), (430, 630), (420, 320)], [(1800, 1000), (1591, 1000), (1230, 1060), (1280, 728), (1297, 317), (840, 300), (420, 320)], [(1230, 1060), (1280, 728), (1297, 317), (840, 300), (420, 320)], [0], [(420, 320), (840, 300), (1297, 317), (1800, 300)]],
[[(1550, 800), (1280, 728), (1297, 317), (1800, 300)], [(1297, 317), (1800, 300)], [(1580, 1200), (1591, 1000), (1800, 1000), (1820, 700), (2011, 600), (1821, 540), (1800, 300)], [(875, 900), (820, 1156), (1230, 1060), (1280, 728), (1297, 317), (1800, 300)], [(400, 1000), (430, 630), (420, 320), (840, 300), (1297, 317), (1800, 300)], [(850, 500), (840, 300), (1297, 317), (1800, 300)], [(2200, 600), (2011, 600), (1821, 540), (1800, 300)], [(1650, 550), (1821, 540), (1800, 300)], [(200, 650), (430, 630), (420, 320), (840, 300), (1297, 317), (1800, 300)], [(650, 650), (430, 630), (420, 320), (840, 300), (1297, 317), (1800, 300)], [(1800, 1000), (1820, 700), (2011, 600), (1821, 540), (1800, 300)], [(1230, 1060), (1280, 728), (1297, 317), (1800, 300)], [(420, 320), (840, 300), (1297, 317), (1800, 300)], [0]]]
    placelist = ['Admin', 'Cafeteria', 'Communcations', 'Electrical', 'Lower engines', 'Medbay', 'Navigation', 'O2',
                 'Reactor', 'Security', 'Shields', 'Storage', 'Upper engines', 'Weapons']
    start_index = placelist.index(start)
    stop_index = placelist.index(stop)
    #st.header(f"{start_index}, {stop_index}")
    #st.header(pathing_matrix[start_index][stop_index])
    return pathing_matrix[start_index][stop_index]


def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n - 1)


def printit(string, firstlist=[], firstnum=314, secondlist=[], secondnum=314, thirdlist=[], thirdnum=314):
    toprint = ""
    for pos in range(len(string)):
        if pos in firstlist:
            toprint += str(firstnum)
        elif pos in secondlist:
            toprint += str(secondnum)
        elif pos in thirdlist:
            toprint += str(thirdnum)
        else:
            toprint += string[pos]
    st.write(toprint)


def code(string):
    #print(string)
    poslistt = []
    numlist = []
    for m in range(len(string)):
        try:
            numlist.append(int(string[m]))
        except:
            poslistt.append(m)
    if poslistt == []:
        st.write("Your code is already solved!")
        return 0
    #print("letter positions are: \n" + str(poslistt))
    #print("numbers unused are:")
    all10 = list(range(10))
    numleft = set(all10) - set(numlist)
    numleftlist = list(numleft)
    #print(numleftlist)
    NL = []
    AL = []
    SL = []
    counter = [0, 0, 0]
    for i in poslistt:
        b = string[i]
        if b == "N":
            NL.append(i)
            counter[0] = 1
        if b == "S":
            SL.append(i)
            counter[1] = 1
        if b == "H":
            AL.append(i)
            counter[2] = 1
    #print(NL, AL, SL)
    variables = sum(counter)
    #print(variables)
    num_to_try = factorial(len(numleftlist)) / factorial(len(numleftlist) - variables)
    st.header("There are " + str(int(num_to_try)) + " possible codes to try:")
    if st.button("Click here if you want to see all possible codes."):
        if counter[0] == 1:
            firstvar = NL
            if counter[1] == 1:
                secondvar = SL
                if counter[2] == 1:
                    thirdvar = AL
            elif counter[2] == 1:
                secondvar = AL
        elif counter[1] == 1:
            firstvar = SL
            if counter[2] == 1:
                secondvar = AL
        else:
            firstvar = AL

        for i in numleftlist:
            if variables == 1:
                printit(string, firstvar, i)
            else:
                secondl = numleftlist.copy()
                secondl.remove(i)
                for j in secondl:
                    if variables == 2:
                        printit(string, firstvar, i, secondvar, j)
                    else:
                        thirdl = secondl.copy()
                        thirdl.remove(j)
                        for k in thirdl:
                            printit(string, firstvar, i, secondvar, j, thirdvar, k)


def combine(string1, string2):
    pairlist = [[string1[i], string2[i]] for i in range(len(string1))]
    for i in range(len(pairlist)):
        for j in [0, 1]:
            if str(pairlist[i][j]).isdigit():
                pairlist[i][j] = int(pairlist[i][j])
    final = [0] * len(string1)
    keydict = {}
    for i in range(len(pairlist)):
        if type(pairlist[i][0]) == type(pairlist[i][1]):
            final[i] = pairlist[i][0]
        else:
            if str(pairlist[i][0]).isdigit():
                keydict[pairlist[i][1]] = pairlist[i][0]
                final[i] = pairlist[i][0]
            elif str(pairlist[i][1]).isdigit():
                keydict[pairlist[i][0]] = pairlist[i][1]
                final[i] = pairlist[i][1]
    for i in range(len(pairlist)):
        if final[i] in keydict.keys():
            final[i] = keydict[final[i]]
    final = ''.join([str(elem) for elem in final])
    return (final)

def print_solution(data, manager, routing, solution, places_list,  point_dict, im, ghostval):
    """Prints solution on console."""
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route_statement = 'The optimal route is:\n'
        plan_output = ""
        times = ""
        route_distance = 0
        best_route = []
        best_times = [0]
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(places_list[manager.IndexToNode(index)])
            best_route.append(places_list[manager.IndexToNode(index)])
            previous_index = index

            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
            short_distance = routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
            times += ' {} -> '.format(short_distance/100)
            best_times.append(short_distance/100)
        d = {'Locations': best_route, 'Times': best_times[:-1]}
        df = pd.DataFrame(data=d)
        plan_output = plan_output[:-4]
        plan_output += '\n\n'
        plan_output += times[:-12] + "\n\n"
        plan_output += 'Time required: {} seconds \n'.format(float(route_distance)/100)
        #st.write(route_statement)
        #st.write(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    draw = ImageDraw.Draw(im)
    prev = (1297, 317)
    ordered_points = []

    for location in best_route:
        ordered_points.append(point_dict[location])
    if ghostval:
    #This is the straight line, no junctions version.
        for i in ordered_points:
            i = i[1:-1]
            i = tuple(map(int, i.split(', ')))
            line = [prev, i]
            draw.line(line, fill=128, width=23)
            prev = i
        st.image(im, caption='Optimal Route', use_column_width=True)
        units = "units"
    else:
        starting_loc = "Cafeteria"
        prev_loc = (1297, 317)

        for i in best_route[1:]:

            list_of_tuples = return_pathing_tuples(starting_loc, i)
            starting_loc = i
            try:
                if list_of_tuples[0] != loctuple:
                    list_of_tuples.reverse()
            except:
                pass
            for loctuple in list_of_tuples:
                line = [prev_loc, loctuple]
                draw.line(line, fill=128, width=23)
                prev_loc = loctuple
        st.image(im, caption='Optimal Route', use_column_width=True)
        units = "seconds"
    if len(df['Locations']) > 1:
        st.write(df[1:])
        st.write(f"This will take {round(df['Times'].sum(), 4)} {units}")

    if not st.sidebar.checkbox("I like Among Us!", value=True, key=None):
        if st.sidebar.checkbox("I like Call of Duty: Warzone better.", value = False):
            if st.sidebar.checkbox("I like looking in stadiums for itty bitty access cards!", value = False):
                st.write("Input your code or codes, with N for the nose symbol, H for the house symbol, \
                         and S for the squiggly symbol that looks like a dollar sign. This will show you the \
                         codes that you should guess.")
                user_code1 = st.text_input("Input your first code", "Example: 4H8N3SHN")
                user_code2 = st.text_input("Input your second code", "")
                user_code3 = st.text_input("Input your third code if you have one", "")
                if not user_code2:
                    finalstring = user_code1
                else:
                    string3 = combine(user_code1, user_code2)
                    if user_code3:
                        finalstring = combine(string3, user_code3)
                    else:
                        finalstring = string3
                st.header("Your code is: " + finalstring)
                code(finalstring)

def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    im = Image.open("el mapa.png")

    placelist = ['Admin', 'Cafeteria', 'Communcations', 'Electrical', 'Lower engines', 'Medbay', 'Navigation', 'O2',
                 'Reactor', 'Security', 'Shields', 'Storage', 'Upper engines', 'Weapons']
    st.title("Among Us Task TSP")
    st.header("By Abraham Holleran")
    task_list = []
    for i in range(len(placelist)):
        if placelist[i] == "Cafeteria":
            task_list.append(True)
        else:
           task_list.append(st.sidebar.checkbox(placelist[i], value=False, key=None))
    task_list.append(True)
    dist_matrix, placelist, pointmatrix, pointdict, ghostval = drop_false(task_list)
    starting = placelist.index("Cafeteria")
    data = create_data_model(dist_matrix, starting)
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['starts'], data['ends'],
                                           )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    if not ghostval:
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            10000,  # vehicle maximum travel distance. Max is 5921 to go to all in my experience.
            True,  # start cumul to zero
            dimension_name)
    else:
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            1000000,  # vehicle maximum travel distance. Max is 5921 to go to all in my experience.
            True,  # start cumul to zero
            dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution, placelist, pointdict, im, ghostval)
        #st.write(task_list)

if __name__ == "__main__":
    main()
