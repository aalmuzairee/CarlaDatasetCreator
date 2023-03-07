import sys
import os
import settings
sys.path.append(settings.CARLA_EGG_PATH)
import carla
import random
import time
import numpy as np

from spawn_npc import NPCClass
from bounding_boxes import create_kitti_datapoint
from set_synchronous_mode import CarlaSyncMode
from bb_filter import apply_filters_to_3d_bb
from WeatherSelector import WeatherSelector
import configs

from data_export import *

import matplotlib.pyplot as plt



class CarlaWorld:
    def __init__(self, output_dir, phase):
        self.output_dir = f"./datasets/kitti/{output_dir}"
        # Carla initialization
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        #self.world = client.load_world("Town01")
        self.world = self.client.get_world()
        print('Successfully connected to CARLA')
        self.blueprint_library = self.world.get_blueprint_library()
        # Sensors stuff
        self.camera_x_location = 0.0
        self.camera_y_location = 0.0
        self.camera_z_location = configs.CAMERA_HEIGHT_POS
        self.sensors_list = []
        # Weather stuff
        self.weather_options = WeatherSelector().get_weather_options()  # List with weather options

        # Recording stuff
        self.total_recorded_frames = 0
        self.first_time_simulating = True

        # data saving
        """ OUTPUT FOLDER GENERATION """
        self.output_folder = os.path.join(self.output_dir, phase)
        folders = ['calib', 'image_2', 'label_2', 'velodyne', 'planes']

        for folder in folders:
            directory = os.path.join(self.output_folder, folder)
            self.maybe_create_dir(directory)

        """ DATA SAVE PATHS """
        self.GROUNDPLANE_PATH = os.path.join(self.output_folder, 'planes/{0:06}.txt')
        self.LIDAR_PATH = os.path.join(self.output_folder, 'velodyne/{0:06}.bin')
        self.LABEL_PATH = os.path.join(self.output_folder, 'label_2/{0:06}.txt')
        self.IMAGE_PATH = os.path.join(self.output_folder, 'image_2/{0:06}.png')
        self.CALIBRATION_PATH = os.path.join(self.output_folder, 'calib/{0:06}.txt')

        self.captured_frame_no = self.current_captured_frame_num()
    
    def maybe_create_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def load_town(self, townName):
        self.world = self.client.load_world(townName)


    def set_weather(self, weather_option):
        # Changing weather https://carla.readthedocs.io/en/stable/carla_settings/
        # Weather_option is one item from the list self.weather_options, which contains a list with the parameters
        weather = carla.WeatherParameters(*weather_option)
        self.world.set_weather(weather)

    def remove_npcs(self):
        print('Destroying actors...')
        self.NPC.remove_npcs()
        print('Done destroying actors.')

    def spawn_npcs(self, number_of_vehicles, number_of_walkers):
        self.NPC = NPCClass()
        self.vehicles_list, _, self.non_players = self.NPC.create_npcs(number_of_vehicles, number_of_walkers)

    def current_captured_frame_num(self):
        # Figures out which frame number we currently are on
        # This is run once, when we start the simulator in case we already have a dataset.
        # The user can then choose to overwrite or append to the dataset.
        label_path = os.path.join(self.output_folder, 'label_2/')
        num_existing_data_files = len(
            [name for name in os.listdir(label_path) if name.endswith('.txt')])
        print(num_existing_data_files)
        if num_existing_data_files == 0:
            return 0
        answer = input(
            "There already exists a dataset in {}. Would you like to (O)verwrite or (A)ppend the dataset? (O/A)".format(
                self.output_folder))
        if answer.upper() == "O":
            logging.info(
                "Resetting frame number to 0 and overwriting existing")
            # Overwrite the data
            return 0
        logging.info("Continuing recording data on frame number {}".format(
            num_existing_data_files))
        return num_existing_data_files

    def put_rgb_sensor(self, vehicle, sensor_width=1248, sensor_height=384, fov=90):
        # https://carla.readthedocs.io/en/latest/cameras_and_sensors/
        bp = self.blueprint_library.find('sensor.camera.rgb')
        # bp.set_attribute('enable_postprocess_effects', 'True')  # https://carla.readthedocs.io/en/latest/bp_library/
        bp.set_attribute('image_size_x', f'{sensor_width}')
        bp.set_attribute('image_size_y', f'{sensor_height}')
        bp.set_attribute('fov', f'{fov}')

        # Adjust sensor relative position to the vehicle
        spawn_point = carla.Transform(carla.Location(x=self.camera_x_location, z=self.camera_z_location))
        self.rgb_camera = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
        self.rgb_camera.blur_amount = 0.0
        self.rgb_camera.motion_blur_intensity = 0
        self.rgb_camera.motion_max_distortion = 0

        # Camera calibration
        calibration = np.identity(3)
        calibration[0, 2] = sensor_width / 2.0
        calibration[1, 2] = sensor_height / 2.0
        calibration[0, 0] = calibration[1, 1] = sensor_width / (2.0 * np.tan(fov * np.pi / 360.0))
        self.rgb_camera.calibration = calibration  # Parameter K of the camera
        self.sensors_list.append(self.rgb_camera)
        return self.rgb_camera

    def put_depth_sensor(self, vehicle, sensor_width=1248, sensor_height=384, fov=90):
        # https://carla.readthedocs.io/en/latest/cameras_and_sensors/
        bp = self.blueprint_library.find('sensor.camera.depth')
        bp.set_attribute('image_size_x', f'{sensor_width}')
        bp.set_attribute('image_size_y', f'{sensor_height}')
        bp.set_attribute('fov', f'{fov}')

        # Adjust sensor relative position to the vehicle
        spawn_point = carla.Transform(carla.Location(x=self.camera_x_location, z=self.camera_z_location))
        self.depth_camera = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
        self.sensors_list.append(self.depth_camera)
        return self.depth_camera

    def put_lidar_sensor(self, vehicle):
        # https://carla.readthedocs.io/en/latest/cameras_and_sensors/
        # keep the lidarconfiguration in configs.py file
        bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        for key, value in configs.lidar_configs.items():
            bp.set_attribute(key, f'{value}')

        # Adjust sensor relative position to the vehicle
        spawn_point = carla.Transform(carla.Location(x=0.0, z=configs.LIDAR_HEIGHT_POS))
        self.lidar_sensor = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
        self.sensors_list.append(self.lidar_sensor)
        return self.lidar_sensor


    def process_depth_data(self, data, sensor_width, sensor_height):
        """
        normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
        in_meters = 1000 * normalized
        """
        data = np.array(data.raw_data)
        data = data.reshape((sensor_height, sensor_width, 4))
        data = data.astype(np.float32)
        # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
        normalized_depth = np.dot(data[:, :, :3], [65536.0, 256.0, 1.0])
        normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
        depth_meters = normalized_depth * 1000
        return depth_meters

    def get_bb_data(self):
        vehicles_on_world = self.world.get_actors().filter('vehicle.*')
        walkers_on_world = self.world.get_actors().filter('walker.*')
        bounding_boxes_vehicles = ClientSideBoundingBoxes.get_bounding_boxes(vehicles_on_world, self.rgb_camera)
        bounding_boxes_walkers = ClientSideBoundingBoxes.get_bounding_boxes(walkers_on_world, self.rgb_camera)
        return [bounding_boxes_vehicles, bounding_boxes_walkers]

    def process_rgb_img(self, img, sensor_width, sensor_height):
        img = np.frombuffer(img.raw_data,  dtype=np.dtype("uint8"))
        img = img.reshape((sensor_height, sensor_width, 4))
        img = img[:, :, :3]  # taking out opacity channel
        # bb = self.get_bb_data()
        return img #, bb

    def generate_datapoints(self, image, depth_image, vehicle, intrinsic_mat, extrinsic_mat):
        """ Returns a list of datapoints (labels and such) that are generated this frame together with the main image
        image """
        datapoints = []
        image = image.copy()
        # Remove this
        rotRP = np.identity(3)
        # Stores all datapoints for the current frames

        for agent in self.non_players:
            image, kitti_datapoint = create_kitti_datapoint(
                agent, intrinsic_mat,extrinsic_mat, image, depth_image, vehicle, rotRP)
            if kitti_datapoint:
                datapoints.append(kitti_datapoint)

        return image, datapoints
    def save_to_files(self, image_data, datapoints, point_cloud, vehicle, intrinsic_mat, extrinsic_mat, captured_frame_no):
        """ Save data in Kitti dataset format """
        logging.info("Attempting to save at frame no: {}".format(captured_frame_no))
        groundplane_fname = self.GROUNDPLANE_PATH.format(captured_frame_no)
        lidar_fname = self.LIDAR_PATH.format(captured_frame_no)
        kitti_fname = self.LABEL_PATH.format(captured_frame_no)
        img_fname = self.IMAGE_PATH.format(captured_frame_no)
        calib_filename = self.CALIBRATION_PATH.format(captured_frame_no)

        save_groundplanes(groundplane_fname, vehicle, configs.LIDAR_HEIGHT_POS)
        save_ref_files(self.output_folder, captured_frame_no)
        save_image_data(img_fname, image_data)
        save_kitti_data(kitti_fname, datapoints)

        save_calibration_matrices(calib_filename, intrinsic_mat, extrinsic_mat)
        save_lidar_data(lidar_fname, point_cloud)

    def remove_sensors(self):
        for sensor in self.sensors_list:
            sensor.destroy()
        self.sensors_list = []




    def begin_data_acquisition(self, sensor_width, sensor_height, fov, frames_to_record_one_ego=10, timestamps=[], egos_to_run=10):
        # Changes the ego vehicle to be put the sensor
        current_ego_recorded_frames = 0

        # No datapoints nearby counter
        no_data_count = 0

        while True: # Loop to change vehicle if no datapoints nearby
            # These vehicles are not considered because the cameras get occluded without changing their absolute position
            ego_vehicle = random.choice([x for x in self.world.get_actors().filter("vehicle.*") if x.type_id not in
                                        ['vehicle.audi.tt', 'vehicle.carlamotors.carlacola', 'vehicle.volkswagen.t2']])
            self.put_rgb_sensor(ego_vehicle, sensor_width, sensor_height, fov)
            self.put_depth_sensor(ego_vehicle, sensor_width, sensor_height, fov)
            self.put_lidar_sensor(ego_vehicle)

            # Begin applying the sync mode
            with CarlaSyncMode(self.world, self.rgb_camera, self.depth_camera, self.lidar_sensor, fps=configs.FPS) as sync_mode:
                # Skip initial frames where the car is being put on the ambient
                if self.first_time_simulating:
                   print("Waiting to start\r",end="")
                   for _ in range(30): #Used to be 30 instead of 10
                       sync_mode.tick_no_data()

                while True: # Loop to keep capturing datapoints
                    if current_ego_recorded_frames == frames_to_record_one_ego:
                        print('\n')
                        self.remove_sensors()
                        return timestamps

                    # Advance the simulation and wait for the data
                    # Skip every nth frame for data recording, so that one frame is not that similar to another
                    wait_frame_ticks = 0
                    while wait_frame_ticks < 10:
                        print("Waiting to change scene a bit\r",end="")
                        sync_mode.tick_no_data()
                        wait_frame_ticks += 1

                    _, rgb_data, depth_data, lidar_data = sync_mode.tick(timeout=2.0)  # If needed, self.frame can be obtained too

                    # Processing raw data
                    image = self.process_rgb_img(rgb_data, sensor_width, sensor_height)
                    intrinsic_mat = self.rgb_camera.calibration
                    extrinsic_mat = self.rgb_camera.get_transform().get_matrix()
                    image, datapoints = self.generate_datapoints(image, depth_data, ego_vehicle, intrinsic_mat, extrinsic_mat)

                    print("Checking for datapoints\r",end="")

                    if datapoints:
                        print("Found datapoints!\r",end="")
                        data = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
                        data = np.reshape(data, (int(data.shape[0] / 4), 4))
                        # Isolate the 3D data
                        points = data[:, :-1]
                        # transform to car space
                        # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
                        # points = np.dot(sync_mode.player.get_transform().get_matrix(), points.T).T
                        # points = points[:, :-1]
                        # points[:, 2] -= LIDAR_HEIGHT_POS

                        #plt.imshow(image)
                        #plt.show()
                        
                        self.save_to_files(rgb_data, datapoints, points, ego_vehicle, intrinsic_mat, extrinsic_mat, self.captured_frame_no)
                        self.captured_frame_no += 1
                    	
                        current_ego_recorded_frames += 1
                        print(f"num_frames captured : {self.captured_frame_no}\r", end="")
                        #break
                    else:
                        no_data_count += 1
                        if no_data_count == 5:
                            no_data_count = 0
                            self.remove_sensors()
                            break



                # depth_array = self.process_depth_data(depth_data, sensor_width, sensor_height)
                # ego_speed = ego_vehicle.get_velocity()
                # ego_speed = np.array([ego_speed.x, ego_speed.y, ego_speed.z])
                # bounding_box = apply_filters_to_3d_bb(bounding_box, depth_array, sensor_width, sensor_height)
                # timestamp = round(time.time() * 1000.0)

                # # Saving into opened HDF5 dataset file
                # self.HDF5_file.record_data(rgb_array, depth_array, bounding_box, ego_speed, timestamp)

                # timestamps.append(timestamp)

                # sys.stdout.write("\r")
                # sys.stdout.write('Frame {0}/{1}'.format(
                #     self.total_recorded_frames, frames_to_record_one_ego*egos_to_run*len(self.weather_options)))
                # sys.stdout.flush()
