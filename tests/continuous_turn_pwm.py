#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side, with a python defined agent"""

import argparse
import collections
import datetime
import logging
import math
import os
import numpy.random as random
import re
import sys
import weakref
import queue

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/PythonAPI/carla')
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/CBF_CLF')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # pylint: disable=import-error

# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from control import get_control

from utils.carla_utils_auto import *

# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================


def steer_to_beta(steer):

    # steer is between [-1, 1] to [-70, 70] degrees
    steer = math.radians(steer*70)
    # kinematic bicycle model
    beta = math.atan((2.102/(2.102+1.553)) * math.tan(steer))

    return beta


def beta_to_steer(beta):

    steer = math.atan(math.tan(beta)*((2.102+1.553)/2.102))

    steer = math.degrees(steer) / 70

    return steer


def get_throttle_brake(ax, vx):

    # use tesla model3

    Cm1 = 7870.5132
    Cm2 = -56.8534
    Cr0 = 1179.1085
    Cr2 = -0.5040

    m = 1845

    numerator = ax * m + Cr0 + Cr2 * vx**2
    denominator = Cm1 - Cm2 * vx
    if denominator == 0:
        print(f"Warning: Division by zero at vx = {vx}")
        return None
    pwm = numerator / denominator

    throttle = max(0, min(pwm, 1))
    brake = min(0, max(pwm, -1))

    return (throttle, brake)


def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """

    pygame.init()
    pygame.font.init()
    world = None
    camera = None
    image_queue = queue.Queue()
    save_path = "../out/continuous_turn/"

    # load LLaVA model
    # processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    # model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit=True)
    # model.to("cuda:0")

    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        if args.sync:

            print("In synchronous mode")
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.02
            sim_world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)

        # chosen start and end location for the test
        spawn_point = carla.Transform(carla.Location(x=-45.11823272705078, y=60.35600280761719, z=0.6000000238418579), carla.Rotation(pitch=0.0, yaw=-92.52740478515625, roll=0.0))
        destination = carla.Transform(carla.Location(x=-104.03401184082031, y=-15.75542163848877, z=0.6000000238418579), carla.Rotation(pitch=0.0, yaw=-97.00296020507812, roll=0.0))

        blueprint_library = client.get_world().get_blueprint_library()
        blueprint = blueprint_library.find('vehicle.tesla.model3')

        args.spawn_point = spawn_point
        args.blueprint = blueprint
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
        # spawn_points = world.map.get_spawn_points()
        # destination = random.choice(spawn_points).location
        agent.set_destination(destination.location)

        # list_actor = world.world.get_actors()
        # for actor_ in list_actor:
        #     if isinstance(actor_, carla.TrafficLight):
        #         actor_.set_state(carla.TrafficLightState.Green) 
        #         actor_.set_green_time(1000.0)

        clock = pygame.time.Clock()

        # camera_init_trans = world.camera_manager._camera_transforms[1][0]
        # camera_bp = world.world.get_blueprint_library().find('sensor.camera.rgb')
        # camera_bp.set_attribute("image_size_x",str(1280))
        # camera_bp.set_attribute("image_size_y",str(720))
        # camera = world.world.spawn_actor(camera_bp, camera_init_trans, attach_to=world.player)
        # camera.listen(image_queue.put)

        # waypoints = world.map.generate_waypoints(1)
        # for w in waypoints:
            # world.world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
            #                                    color=carla.Color(r=255, g=0, b=0), life_time=120.0,
            #                                    persistent_lines=True)

        # planned_path = agent.get_local_planner()._waypoints_queue

        # Display waypoints in the trajectory
        # for wp in planned_path:
        #     print(f"Planned Waypoint at {wp[0].transform.location}")
        #     world.world.debug.draw_string(wp[0].transform.location, 'O', draw_shadow=False,
        #                                        color=carla.Color(r=255, g=0, b=0), life_time=30.0,
        #                                        persistent_lines=True)

        count = 0
        legnth = None
        acc = None
        beta = None

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
            
            if count == 0:
                # get reference control from simulator
                steer_ref = control.steer
                print("steer_ref", steer_ref)
                beta_ref = steer_to_beta(steer_ref)
                accel = world.player.get_acceleration()
                yaw = math.radians(world.player.get_transform().rotation.yaw)
                acc_ref = accel.x * math.cos(yaw) + accel.y * math.sin(yaw)
                u_ref = np.array([acc_ref, beta_ref])

                # get x0 current state
                transform = world.player.get_transform()  # Get transform (location & rotation)
                location = transform.location  # Extract location
                velocity = world.player.get_velocity()

                x = location.x
                y = location.y
                theta = yaw
                # v = (velocity.x * math.cos(yaw) + velocity.y * math.sin(yaw)) * 3.6
                v = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)

                print(x,y,theta, v)

                x0 = np.array([[x, y, theta, v]]).T

                params = {
                    # 'obstacle': {
                    #     'xo':-45,
                    #     'yo':48,
                    #     'rsafe':4
                    # },
                    'speed':{
                        'target_speed': 15
                    }
                }

                ut, time_steps = get_control(x0, params, u_ref)
                acc = ut[0,:]
                beta = ut[1,:]
                length = len(acc)

                # for i in range(10):
                #     print(beta[i])
                #     print(acc[i])

            velocity = world.player.get_velocity()
            acceleration = world.player.get_acceleration()

            # Compute longitudinal (forward) velocity
            vx = (velocity.x * math.cos(yaw) + velocity.y * math.sin(yaw)) * 3.6
            
            throttle, brake = get_throttle_brake(acc[count], vx)

            steer = beta_to_steer(beta[count])
            print(f"throttle {throttle}, brake {brake}, steer {steer}")

            control.throttle = throttle
            control.brake = brake
            control.steer = steer

            
            if count == length - 1:
                count = 0
            else:
                count += 1

            control.manual_gear_shift = False
            world.player.apply_control(control)


    finally:

        # print('saving images...')
        # image_list = list(image_queue.queue)
        # queue_size = len(image_list)
        # selected_images = [image_list[i] for i in range(0, queue_size, 10)]
        # for image in selected_images:
        #     image.save_to_disk(save_path + '%08d' % image.frame)

        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            world.destroy()
        
        if camera is not None:
            camera.destroy()

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
        '--sync', default=True,
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
