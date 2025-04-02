import numpy as np
import os
import sys
import math

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/PythonAPI/carla')
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/CBF_CLF_toolbox')
except:
    pass

import kinematic_bicycle as kb
import ode_solver
from drive_modules import straight, change_speed, turn, change_lane

from utils.unit_converter import *
from utils.carla_utils_manual import *


def get_state(player):

    transform = player.get_transform()  # Get transform (location & rotation)
    location = transform.location  # Extract location
    velocity = player.get_velocity()
    yaw = math.radians(player.get_transform().rotation.yaw)

    x = location.x
    y = location.y
    theta = yaw
    v = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)

    return np.array([x, y, theta, v])


def distance(a,b):

    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None

    client = carla.Client(args.host, args.port)
    client.set_timeout(2000.0)

    # change init args here
    args.sync = True
  
    spawn_point = carla.Transform(carla.Location(x=-66.99111938476562, y=16.450407028198242, z=0.6000000238418579), carla.Rotation(pitch=-10.414794921875, yaw=179.5846405029297, roll=0.0012027793563902378))
    
    blueprint_library = client.get_world().get_blueprint_library()
    blueprint = blueprint_library.find('vehicle.tesla.model3')

    args.spawn_point = spawn_point
    args.blueprint = blueprint

    start = carla.Location(x=-66.99111938476562, y=16.450407028198242, z=1.5217187404632568)
    end = carla.Location(x=-82.24272918701172, y=13.077672004699707, z=2.1280815601348877)


    try:
        
        sim_world = client.get_world()
        traffic_manager = client.get_trafficmanager()
        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.02
            sim_world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                  "experience some issues with the traffic simulation")

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)

    
        world = World(sim_world, hud, traffic_manager, args)
        controller = KeyboardControl(world, args.autopilot)

        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()

        module = change_lane(start, end, world.world)

        while True:
            if args.sync:
                sim_world.tick()
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock, args.sync):
                return

            # control from keyboard
            control = controller._control


            steer_ref = control.steer
            # print("steer_ref", steer_ref)
            beta_ref = steer_to_beta(steer_ref)
            acc_ref = get_acc(control.throttle, control.brake)
            u_ref = np.array([acc_ref, beta_ref])

            x0 = get_state(world.player)
        
            u = module.solve(x0, u_ref)

            u_safe = carla.VehicleControl()



            u_safe.throttle, u_safe.brake = get_throttle_brake(u[0])
            u_safe.steer = beta_to_steer(u[1])

            # print(f"steer_ref: {steer_ref}, beta_ref: {beta_ref}, beta: {u[1]}, steer: {u_safe.steer}")

            world.player.apply_control(u_safe)

            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:

        if original_settings:
            sim_world.apply_settings(original_settings)

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose', action='store_true', dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host', metavar='H', default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port', metavar='P', default=2000, type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot', action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res', metavar='WIDTHxHEIGHT', default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter', metavar='PATTERN', default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation', metavar='G', default='All',
        help='restrict to certain actor generation (values: "2","3","All" - default: "All")')
    argparser.add_argument(
        '--rolename', metavar='NAME', default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma', default=1.0, type=float,
        help='Gamma correction of the camera (default: 1.0)')
    argparser.add_argument(
        '--sync', action='store_true',
        help='Activate synchronous mode execution')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()