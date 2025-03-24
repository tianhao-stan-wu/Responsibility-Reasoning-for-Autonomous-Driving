import carla
import time

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
# world = client.get_world()
world = client.load_world('Town04')

actors = world.get_actors()

vehicles = actors.filter("vehicle.*")

# Destroy all vehicles
for vehicle in vehicles:
    vehicle.destroy()
    print(f"Destroyed vehicle: {vehicle.id}")

# Set synchronous mode for consistent simulation steps
settings = world.get_settings()
settings.synchronous_mode = True  # Enable sync mode
settings.fixed_delta_seconds = 0.05  # Simulation step of 0.05s
world.apply_settings(settings)

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
throttle = 1
steer = 1
brake = 1

try:
    for _ in range(100):  # Run for 100 frames (~5 seconds)
        world.tick()  # Advance simulation step

        # Apply increasing throttle
        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steer
        vehicle.apply_control(control)

        # Get acceleration from CARLA API
        acceleration = vehicle.get_acceleration().length()

        # Print results
        print(f"Throttle: {throttle:.2f}, Acceleration: ({acceleration:.2f}) m/s²")

        # Gradually increase throttle
        # throttle = min(1.0, throttle + throttle_increment)

        time.sleep(0.05)

finally:
    print("Stopping simulation...")
    vehicle.destroy()
    settings.synchronous_mode = False 
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
