import carla

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Get the world and spawn a vehicle
world = client.get_world()

spawn_points = world.get_map().get_spawn_points()

blueprint_library = world.get_blueprint_library()
blueprint = blueprint_library.find('vehicle.tesla.model3')

vehicle = world.try_spawn_actor(blueprint, spawn_points[0])

# Get physics control data
physics_control = vehicle.get_physics_control()

# Extract center of mass
com = physics_control.center_of_mass

# Extract wheel positions
wheels = physics_control.wheels
front_left = wheels[0].position
rear_left = wheels[2].position

# Compute front and rear axle distances
lf = abs(front_left.x - com.x)  # Front axle to CoM
lr = abs(rear_left.x - com.x)   # Rear axle to CoM

print(f"Front axle distance to CoM: {lf} meters")
print(f"Rear axle distance to CoM: {lr} meters")
