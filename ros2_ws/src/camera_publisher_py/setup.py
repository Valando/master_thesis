from setuptools import setup

package_name = 'camera_publisher_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pi',
    maintainer_email='pi@todo.todo',
    description='Camera publisher node',
    license='TODO: License declaration',
     entry_points={
    'console_scripts': [
        'camera_publisher = camera_publisher_py.camera_publisher:main','camera_subscriber = camera_publisher_py.camera_subscriber:main'
    ],
},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),  # This adds the marker to the package index
        ('share/' + package_name, ['package.xml']),  # This installs the package.xml
    ],
)