"""
Alan Naoto
Created: 14/10/2019

Dataset creation plan
200 frames = 1,7 GB
Total planned: 20.000 frames = 170 GB
5 Towns
5 Weathers
x frames per ego vehicle
y amount of ego vehicles

Total frames planned = x * y * 5 weathers * 5 towns
20.000 / 5 towns = x * y * 5 weathers
x * y = 800
if x = 60 frames per ego, then
60 * y = 800
y ~= 13 egos

i.e., 13 egos * 60 frames * 5 weathers = 3900 frames per town
3900 * 5 towns = 19500 frames total
19500 frames ~= 165.75 GB

Suggested amount of vehicles and walkers so that traffic jam occurence is minimized
Town01 - 100 vehic 200 walk
Town02 - 50 vehic 100 walk
Town03 - 200 vehic 150 walk
Town04 - 250 vehic 100 walk
Town05 - 150 vehic 150 walk
"""

import argparse
import os
import sys
from CarlaWorld import CarlaWorld
#from utils.create_video_on_hdf5.create_content_on_hdf5 import read_hdf5_test, treat_single_image, create_video_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settings for the data capture", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('output_dir', default=None, type=str, help='name of output folder to save the data')
    parser.add_argument('-wi', '--width', default=1920, type=int, help="camera rgb and depth sensor width in pixels")
    parser.add_argument('-he', '--height', default=1080, type=int, help="camera rgb and depth sensor width in pixels")
    parser.add_argument('-ve', '--vehicles', default=120, type=int, help="number of vehicles to spawn in the simulation")
    parser.add_argument('-wa', '--walkers', default=100, type=int, help="number of walkers to spawn in the simulation")
    parser.add_argument('-d', '--depth', action='store_true', help="show the depth video side by side with the rgb")
    parser.add_argument('-p', '--phase', default="training", type=str, help="training or testing phase")
    args = parser.parse_args()

    assert(args.width > 0 and args.height > 0)
    if args.vehicles == 0 and args.walkers == 0:
        print('Are you sure you don\'t want to spawn vehicles and pedestrians in the map?')

    # Sensor setup (rgb and depth share these values)
    # 1024 x 768 or 1920 x 1080 are recommended values. Higher values lead to better graphics but larger filesize
    sensor_width = args.width
    sensor_height = args.height
    # Camera and depth field of view, lidar field of view is set in configs
    fov = 90

    # For waymo
    #sensor_width = 1920
    #sensor_height = 1080
    #fov = 50.4
    
    # Beginning data capture proccedure
    phase = "training"
    allTowns = ["Town01","Town02","Town03","Town04","Town05","Town06","Town07","Town10HD"]
    
    
    townNames = ["Town01"]


    CarlaWorld = CarlaWorld(args.output_dir, args.phase)

    for townName in townNames:
        CarlaWorld.load_town(townName)
        print(townName[-1], "/" , len(allTowns), ":",townName)
        timestamps = []
        egos_to_run = 10
        print('Starting to record data...')
        CarlaWorld.spawn_npcs(number_of_vehicles=args.vehicles, number_of_walkers=args.walkers)
        
        weatherNum = 1
        allWeathers = len(CarlaWorld.weather_options)
        
        for weather_option in CarlaWorld.weather_options:
            print("Weather:", weatherNum , "/" , allWeathers)
            CarlaWorld.set_weather(weather_option)
            ego_vehicle_iteration = 0
            while ego_vehicle_iteration < egos_to_run:
                print(f"num_ego_vehicle: {ego_vehicle_iteration}")
                CarlaWorld.begin_data_acquisition(sensor_width, sensor_height, fov,
                                                frames_to_record_one_ego=60, timestamps=timestamps,
                                                egos_to_run=egos_to_run)
                print('Setting another vehicle as EGO.')
                ego_vehicle_iteration += 1
            weatherNum += 1

        CarlaWorld.remove_npcs()
    print('Finished simulation.')
    print('Saving timestamps...')
