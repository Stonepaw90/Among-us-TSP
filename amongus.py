#Coded by Abraham Holleran in 2020 when I was learning Python and didn't know about python classes

import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw

from ortools.constraint_solver import routing_enums_pb2, pywrapcp

from distance_matrices import PATHING_MATRIX, GHOST_DISTANCE_MATRIX, ALIVE_DISTANCE_MATRIX

st.sidebar.button("Re Run")
ghost = st.sidebar.checkbox("Can you pass through walls?")

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
    dmatrix = np.array(GHOST_DISTANCE_MATRIX, dtype = object) if ghost else np.array(ALIVE_DISTANCE_MATRIX)
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
    placelist = ['Admin', 'Cafeteria', 'Communcations', 'Electrical', 'Lower engines', 'Medbay', 'Navigation', 'O2',
                 'Reactor', 'Security', 'Shields', 'Storage', 'Upper engines', 'Weapons']
    start_index = placelist.index(start)
    stop_index = placelist.index(stop)
    #st.header(f"{start_index}, {stop_index}")
    #st.header(pathing_matrix[start_index][stop_index])
    return PATHING_MATRIX[start_index][stop_index]


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

def print_solution(data, manager, routing, solution, places_list,  point_dict, im, ghostval):
    """Prints solution on console."""
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
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
    metric = "Distances" if ghostval else "Times"
    d = {'Locations': best_route, metric: best_times[:-1]}
    if len(best_route) > 1:
        df = pd.DataFrame(data=d)
        st.dataframe(d)
        st.write(f"This will take {round(df[metric].sum(), 4)} {units}")



def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    im = Image.open("el mapa.png")

    placelist = ['Admin', 'Cafeteria', 'Communcations', 'Electrical', 'Lower engines', 'Medbay', 'Navigation', 'O2',
                 'Reactor', 'Security', 'Shields', 'Storage', 'Upper engines', 'Weapons']
    st.title("Among Us Optimal Route Finder")
    st.header("Coded By Abraham Holleran :sunglasses:")
    task_list = []
    for i in range(len(placelist)):
        if placelist[i]  == "Cafeteria":
            task_list.append(True)
        elif placelist[i] in ["Admin", "Upper engines", "Lower engines"]:
            task_list.append(st.sidebar.checkbox(placelist[i], value=True))
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