import os
from glob import glob

from setuptools import find_packages, setup

package_name = "fl_robots"

# Collect web template and static files for data_files
web_data_files = []
web_base = os.path.join(package_name, "web")
for dirpath, dirnames, filenames in os.walk(web_base):
    if filenames:
        install_dir = os.path.join("share", package_name, dirpath)
        files = [os.path.join(dirpath, f) for f in filenames]
        web_data_files.append((install_dir, files))

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test", "test.*", "ROS", "ROS.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Launch files
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
        # Config files
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.yaml"))),
    ]
    + web_data_files,
    # Include package data (web templates and static files)
    package_data={
        package_name: [
            "web/templates/*.html",
            "web/static/css/*.css",
            "web/static/js/*.js",
        ],
    },
    py_modules=[],
    install_requires=[
        "setuptools",
        "pydantic>=2.0",
        "torch",
        "numpy",
        "scikit-learn",
        "flask",
        "flask-cors",
        "flask-socketio",
    ],
    zip_safe=True,
    maintainer="Developer",
    maintainer_email="developer@fl-robots.local",
    description="Federated Learning Multi-Robot Coordination System for ROS2",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "robot_agent = fl_robots.robot_agent:main",
            "aggregator = fl_robots.aggregator:main",
            "coordinator = fl_robots.coordinator:main",
            "monitor = fl_robots.monitor:main",
            "digital_twin = fl_robots.digital_twin:main",
            "web_dashboard = fl_robots.web_dashboard:main",
        ],
    },
)
