import os
from pathlib import Path

from setuptools import find_packages, setup

package_name = "fl_robots"

# Collect web template and static files for data_files
web_data_files = []
web_base = Path(package_name) / "web"
for dirpath_str, _dirnames, filenames in os.walk(str(web_base)):
    if filenames:
        dirpath = Path(dirpath_str)
        install_dir = str(Path("share") / package_name / dirpath)
        files = [str(dirpath / f) for f in filenames]
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
            str(Path("share") / package_name / "launch"),
            [str(p) for p in sorted(Path("launch").glob("*launch.[pxy][yma]*"))],
        ),
        # Config files
        (
            str(Path("share") / package_name / "config"),
            [str(p) for p in sorted(Path("config").glob("*.yaml"))],
        ),
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
