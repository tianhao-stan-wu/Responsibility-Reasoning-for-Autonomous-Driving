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
    sampling_resolution = 1

    grp = GlobalRoutePlanner(amap, sampling_resolution)

    w1 = grp.trace_route(start, end) # there are other funcations can be used to generate a route in GlobalRoutePlanner.
    i = 0
    for w in w1:

        print(w[0].transform.location.x, w[0].transform.location.y)

        if i % 10 == 0:
            world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
            color=carla.Color(r=255, g=0, b=0), life_time=30.0,
            persistent_lines=False)
        else:
            world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
            color = carla.Color(r=0, g=0, b=255), life_time=30.0,
            persistent_lines=False)
        i += 1


client = carla.Client("localhost", 2000)
client.set_timeout(10)
world = client.get_world()


start = carla.Location(x=-66.99111938476562, y=16.450407028198242, z=1.5217187404632568)
end = carla.Location(x=-82.24272918701172, y=13.077672004699707, z=2.1280815601348877)


get_waypoints(world, start, end)