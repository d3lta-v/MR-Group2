from setuptools import setup
import os
from glob import glob

package_name = 'motion_pid_controller'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='PID controller for robot motion control',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'start_pid = motion_pid_controller.motion_pid_controller:main',
        ],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
         (os.path.join('share', package_name), glob('launch/*.launch.py')),
        ('share/' + package_name, ['package.xml']),
    ],
)
