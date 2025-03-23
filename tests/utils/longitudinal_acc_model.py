import csv
import sys
import os

# sys.path.append("/home/twu/Desktop/Responsibility-Reasoning-for-Autonomous-Driving/PythonAPI/carla")
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/PythonAPI/carla')
except:
    print("fail to add path")

from carla_utils_auto import *


def save_data(csv_filename, acc_x, v_x, pwm):
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([acc_x, v_x, pwm])


def get_pwm(throttle, brake):
    
    if throttle > 0:
        return throttle
    elif brake > 0:
        return -1 * brake
    else:
        return 0


def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """

    pygame.init()
    pygame.font.init()
    world = None

    # modify args here
    args.sync = True
    args.filter = "vehicle.tesla.model3"
    args.loop = True

    # initialization
    csv_filename = "data/pwm.csv"
    Frx = []
    v_x = []
    pwm = []

    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["acc_x", "v_x", "pwm"]) 

    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)
        if args.agent == "Basic":
            agent = BasicAgent(world.player, 30)
            agent.follow_speed_limits(True)
        elif args.agent == "Constant":
            agent = ConstantVelocityAgent(world.player, 30)
            ground_loc = world.world.ground_projection(world.player.get_location(), 5)
            if ground_loc:
                world.player.set_location(ground_loc.location + carla.Location(z=0.01))
            agent.follow_speed_limits(True)
        elif args.agent == "Behavior":
            agent = BehaviorAgent(world.player, behavior=args.behavior)

        # Set the agent destination
        spawn_points = world.map.get_spawn_points()
        destination = random.choice(spawn_points).location
        agent.set_destination(destination)

        clock = pygame.time.Clock()

        physics_control = world.player.get_physics_control()

        # Extract mass
        mass = physics_control.mass
        print("mass", mass)

        while True:
            clock.tick()
            if args.sync:
                world.world.tick()
            else:
                world.world.wait_for_tick()
            if controller.parse_events():
                return

            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            if agent.done():
                if args.loop:
                    agent.set_destination(random.choice(spawn_points).location)
                    world.hud.notification("Target reached", seconds=4.0)
                    print("The target has been reached, searching for another target")
                else:
                    print("The target has been reached, stopping the simulation")
                    break

            control = agent.run_step()
            control.manual_gear_shift = False
            world.player.apply_control(control)

            # check x axis aligns with the vehicle's heading
            transform = world.player.get_transform()
            yaw = math.radians(transform.rotation.yaw)  # Convert yaw to radians

            # Get global velocity and acceleration
            velocity = world.player.get_velocity()
            acceleration = world.player.get_acceleration()

            # Compute longitudinal (forward) velocity
            v_x = velocity.x * math.cos(yaw) + velocity.y * math.sin(yaw)

            # Compute longitudinal acceleration
            acc_x = acceleration.x * math.cos(yaw) + acceleration.y * math.sin(yaw)

            pwm = get_pwm(control.throttle, control.brake)

            print("acc_x", acc_x)
            print("v_x", v_x)
            print()

            save_data(csv_filename, acc_x, v_x, pwm)


    finally:

        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose', action='store_true', dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host', metavar='H', default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port', metavar='P', default=2000, type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res', metavar='WIDTHxHEIGHT', default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync', action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter', metavar='PATTERN', default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation', metavar='G', default='All',
        help='restrict to certain actor generation (values: "2","3","All" - default: "All")')
    argparser.add_argument(
        '-l', '--loop', action='store_true', dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str, choices=["Behavior", "Basic", "Constant"], default="Behavior",
        help="select which agent to run")
    argparser.add_argument(
        '-b', '--behavior', type=str, choices=["cautious", "normal", "aggressive"], default='normal',
        help='Choose one of the possible agent behaviors (default: normal)')
    argparser.add_argument(
        '-s', '--seed', default=None, type=int,
        help='Set seed for repeating executions (default: None)')

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

