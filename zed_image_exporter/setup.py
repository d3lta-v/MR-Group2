from setuptools import setup

package_name = 'zed_image_exporter'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Exports ZED images from a rosbag every N frames',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'export_images = zed_image_exporter.export_images:main',
        ],
    },
)
