import carla


"""
Front axle distance to CoM: 1.553 meters
Rear axle distance to CoM: 2.102 meters
"""

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Get the world and spawn a vehicle
world = client.get_world()

spawn_points = world.get_map().get_spawn_points()

blueprint_library = world.get_blueprint_library()
blueprint = blueprint_library.find('vehicle.tesla.model3')

vehicle = world.try_spawn_actor(blueprint, spawn_points[0])

if vehicle is not None:
    # Get physics control data
    physics_control = vehicle.get_physics_control()

    # Transform center of mass to world coordinates
    vehicle_transform = vehicle.get_transform()
    com_world = vehicle_transform.transform(physics_control.center_of_mass)
    print("com_world:", com_world)

    # Extract wheel positions (already in world coordinates)
    wheels = physics_control.wheels
    front_left = wheels[0].position / 100  # Convert from cm to meters
    print("front_left:", front_left)
    rear_left = wheels[2].position / 100   # Convert from cm to meters
    print("rear_left:", rear_left)

    lf = com_world.distance(front_left)
    lr = com_world.distance(rear_left)

    print(f"Front axle distance to CoM: {lf:.3f} meters")
    print(f"Rear axle distance to CoM: {lr:.3f} meters")

    vehicle.destroy()
else:
    print("Failed to spawn vehicle")
