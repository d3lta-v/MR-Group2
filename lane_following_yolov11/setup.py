from setuptools import setup
import os
from glob import glob

package_name = 'lane_following_yolov11'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Lane following and cone mapping using YOLOv11',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'lane_segmentation_node = lane_following_yolov11.yolov11_lane_segmentation_node:main',
        'lane_controller_node = lane_following_yolov11.lane_following_controller_node:main',
        'cone_localization_node = lane_following_yolov11.cone_localization_node:main',
        'autonomous_cone_navigation_node = lane_following_yolov11.autonomous_cone_navigation_node:main',
    ],
},
    
)