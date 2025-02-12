import carla
import random

# connect to server
client = carla.Client('localhost', 2000)
world = client.get_world()

# select map
client.load_world('Town05')

# spawn vehicles
vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
spawn_points = world.get_map().get_spawn_points()

# Spawn 50 vehicles randomly distributed throughout the map 
# for each spawn point, we choose a random vehicle from the blueprint library
for i in range(0,50):
    world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))

# spawn ego vehicle of a specific model
ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
ego_bp.set_attribute('role_name', 'hero')
ego_vehicle = world.spawn_actor(ego_bp, random.choice(spawn_points))

# or choose a random model
# ego_vehicle = world.spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))

# Create a transform to place the camera on top of the vehicle
camera_init_trans = carla.Transform(carla.Location(z=1.5))

# We create the camera through a blueprint that defines its properties
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

# We spawn the camera and attach it to our ego vehicle
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

# set autopilot for other vehicles
for vehicle in world.get_actors().filter('*vehicle*'):
    vehicle.set_autopilot(True)

# set synchronomous mode
settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# must set traffic manager to sync as well
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)



"""
reference code for future use
"""

# carla.Actor mostly consists of get() and set() methods to manage the actors around the map.
print(actor.get_acceleration())
print(actor.get_velocity())

# Actors are not destroyed when a Python script finishes. They have to explicitly destroy themselves.
destroyed_sucessfully = actor.destroy() # Returns True if successful







