from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'fl_robots'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # Include config files
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=[
        'setuptools',
        'torch',
        'numpy',
        'scikit-learn',
    ],
    zip_safe=True,
    maintainer='Developer',
    maintainer_email='your-email@example.com',
    description='Federated Learning Multi-Robot Coordination System for ROS2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_agent = fl_robots.robot_agent:main',
            'aggregator = fl_robots.aggregator:main',
            'coordinator = fl_robots.coordinator:main',
            'monitor = fl_robots.monitor:main',
            'digital_twin = fl_robots.digital_twin:main',
            'web_dashboard = fl_robots.web_dashboard:main',
        ],
    },
)
