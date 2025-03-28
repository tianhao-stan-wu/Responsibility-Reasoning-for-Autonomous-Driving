import carla
import time

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

actors = world.get_actors()

vehicles = actors.filter("vehicle.*")

# Destroy all vehicles
for vehicle in vehicles:
    vehicle.destroy()
    print(f"Destroyed vehicle: {vehicle.id}")

# Spawn Tesla Model 3 at a chosen location
spawn_point = carla.Transform(
    carla.Location(x=-45.118, y=60.356, z=0.6),
    carla.Rotation(pitch=0.0, yaw=-92.527, roll=0.0)
)

blueprint_library = world.get_blueprint_library()
blueprint = blueprint_library.find('vehicle.tesla.model3')
vehicle = world.try_spawn_actor(blueprint, spawn_point)

if vehicle is None:
    print("Failed to spawn vehicle")
    exit()

# Start simulation loop
steering = 0.0  # Max right turn

try:
  

    # Apply control with constant steering
    control = carla.VehicleControl()
    control.steer = steering
    vehicle.apply_control(control)

    # Print results
    print(f"Steering: {steering:.2f}")

finally:
    print("finished")