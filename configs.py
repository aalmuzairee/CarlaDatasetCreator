""" Data Generation Configs"""

""" RENDERING SETTINGS """
# WINDOW_WIDTH = 1248
# WINDOW_HEIGHT = 384
#Waymo
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080

#possible
#WINDOW_WIDTH = 640
#WINDOW_HEIGHT = 360


MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180

WINDOW_WIDTH_HALF = WINDOW_WIDTH / 2
WINDOW_HEIGHT_HALF = WINDOW_HEIGHT / 2

MAX_RENDER_DEPTH_IN_METERS = 70  # Meters
MIN_VISIBLE_VERTICES_FOR_RENDER = 4
MIN_BBOX_AREA_IN_PX = 100

OCCLUDED_VERTEX_COLOR = (255, 0, 0)
VISIBLE_VERTEX_COLOR = (0, 255, 0)

""" Carla Settings """
CAMERA_HEIGHT_POS = 2.0
LIDAR_HEIGHT_POS = CAMERA_HEIGHT_POS

FPS = 10

# For kitti : Common used Velodyne HDL-64E S3 is from -24.33 to 2, with 64 beams and around 102,000 points in real world
# Rotation frequency == FPS to be able to rotate full once
# For waymo: guessing on tutorial 5 waymo lidars, the top is from -0.3149 to 0.0399, all lidar have 126,000 points,
#           with top corresponding to 108,000, and top range is 75 meters while other four are 20 meters

""" Lidar Settings """
# lidar_configs= {'channels' : 40,
#                 'range' : MAX_RENDER_DEPTH_IN_METERS,
#                 'points_per_second' : 720000,
#                 'rotation_frequency' : 10,
#                 'upper_fov' : 2,
#                 'lower_fov': -26.8 }

""" Lidar Settings  for Waymo """
lidar_configs= {'channels' : 32,
                'range' : 75,
                'points_per_second' : 3200000,
                'rotation_frequency' : 10,
                'upper_fov' : 2.4,
                'lower_fov': -17.6, 
                'noise_stddev': 0.2}


