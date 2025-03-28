import carla


def set_location_from_spectator(world):

	spectator = world.get_spectator()
	transform = spectator.get_transform()
	location = transform.location
	rotation = transform.rotation

	print("Spectator:")
	print(f"Location: x={location.x}, y={location.y}, z={location.z}")
	print(f"Rotation: pitch={rotation.pitch}, yaw={rotation.yaw}, roll={rotation.roll}")
	print()

	spawn_point = carla.Transform(carla.Location(x=location.x, y=location.y, z=location.z), carla.Rotation(pitch=0.000000, yaw=rotation.yaw, roll=0.000000))
	print("Spawn point:")
	print(f"Location: x={spawn_point.location.x}, y={spawn_point.location.y}, z={spawn_point.location.z}")
	print(f"Rotation: pitch={spawn_point.rotation.pitch}, yaw={spawn_point.rotation.yaw}, roll={spawn_point.rotation.roll}")

	return spawn_point


def get_tesla_blueprint(world):

    blueprint_library = world.get_blueprint_library()
    blueprint = blueprint_library.find('vehicle.tesla.model3')

    return blueprint





client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
set_location_from_spectator(client.get_world())