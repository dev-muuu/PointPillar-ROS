from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'pointpillars_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        (os.path.join('lib', package_name), glob('scripts/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Indra',
    maintainer_email='dengyinan@bit.edu.cn',
    description='The pointpillars_ros package for ROS2',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
