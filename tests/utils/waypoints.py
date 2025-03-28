import carla
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/PythonAPI/carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner
# from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO


def get_waypoints(world, start, end):

    '''
    start, end are carla locations specifying the starting and ending coordinates
    '''

    amap = world.get_map()
    sampling_resolution = 2

    grp = GlobalRoutePlanner(amap, sampling_resolution)

    w1 = grp.trace_route(start, end) # there are other funcations can be used to generate a route in GlobalRoutePlanner.
    i = 0
    for w in w1:
        if i % 10 == 0:
            world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
            color=carla.Color(r=255, g=0, b=0), life_time=10.0,
            persistent_lines=False)
        else:
            world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
            color = carla.Color(r=0, g=0, b=255), life_time=10.0,
            persistent_lines=False)
        i += 1


client = carla.Client("localhost", 2000)
client.set_timeout(10)
world = client.get_world()


start = carla.Location(x=-45.11823272705078, y=60.35600280761719, z=0.6000000238418579)
end = carla.Location(x=-45.11823272705078, y=-20.35600280761719, z=0.6000000238418579)


get_waypoints(world, start, end)