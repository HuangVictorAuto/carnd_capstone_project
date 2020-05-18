## Self Driving Car Capstone Project

### Overview:

This is the capstone project for the Udacity Self Driving Car Nanodegree. The final project integrated the essential parts of a self driving car system: perception, planning and control. The system is developed based on ROS(Robotic Operating System). The goal is to run the car autonomously on the test track in simulator and on the real car.

As this is a individual submission. The code will only work for ***Highway in Simulator.*** 

### Individual Submission Info:

Name: Huang Victor

Email: hitvictor_wisdom@163.com



### Software Architecture

For a self driving car, the system usually have the following subsystem working together.

#### Sensor:

Sensors are the hardware that help the car to obtain the environment information and self current state. The commonly used sensors are, Radar, Lidar, Camera, Ultrosonic, IMU, GPS and so on.

For this capstone project, we focus on the **Sensor Camera**.

#### Perception:

The perception system is to transfer the raw sensor data into structured useful information that can be used for later planning and control. The perception subsystem normally can be divided into detection and localization subsystems. 

 The detection part is to understand the surrounding environment, like lane detection, traffic sign and traffic light detection and classification, passenger or object detection and tracking.

The localization part is to tell where the car is in a map.

For this capstone project, we focus on processing the camera image to construct a **Traffic Light Detector and Classifier.**

#### Planning:

The planning system is to plan how the car move from A to B in a map with all the perception information. The high level is called route planning, it plan a path from point A to point B in a map like what Google Map offers. The low level include the behavior planning and trajectory planning. The trajectory is to generate a timelized path for the car to follow.

For this capstone project, we focus on the **Trajectory Planning**.  Generate the corresponding trajectory according to the traffic light information.

#### Control:

The Control system control the steer, throttle and brake of the car to make sure the car follow the planned trajectory steady, quickly and precisely. Normally the system include a PID Contorller, MPC Controller etc.

For this capstone project, we focus on a **PID Controller** to transfer the planned trajectory to throttle and brake.

#### ROS Architecture

The different subsystem are structured as ROS Node. And different systems communicate with each other through publish and subscribe to a specific ROS Topic in the ROS Message form. ![](imgs\ROSNodes_Architecture.png)

In this project the following packages have been built or modified to navigate the vehicle around the test track.

##### Traffic Light Detection:

![](imgs\TLD_Node.png)

This package contains the traffic light detection node: `tl_detector.py`. This node takes in data from the `/image_color`, `/current_pose`, and `/base_waypoints` topics and publishes the locations to stop for red traffic lights to the `/traffic_waypoint topic`.

The `/current_pose` topic provides the vehicle's current position, and `/base_waypoints` provides a complete list of waypoints the car will be following.

Built both traffic light detection node and a traffic light classification node. Traffic light detection was implemented in `tl_detector.py` and traffic light classification was implemented at `../tl_detector/light_classification_model/tl_classfier.py`.

##### Waypoint Updater:

![](imgs\WaypointUpdater_Node.png)

This package contains the waypoint updater node: `waypoint_updater.py`. The purpose of this node is to update the target velocity property of each waypoint based on traffic light and obstacle detection data. This node subscribes to the `/base_waypoints`, `/current_pose`, `/obstacle_waypoint`, and `/traffic_waypoint` topics, and publishes a list of waypoints ahead of the car with target velocities to the `/final_waypoints` topic

##### DBW Node:

![](imgs\DBW_Node.png)

Carla is equipped with a drive-by-wire (dbw) system, meaning the throttle, brake, and steering have electronic control. This package consists of files responsible for control of the vehicle: the node `dbw_node.py` and the file `twist_controller.py`, along with a pid and lowpass filter. The `dbw_node` subscribes to the `/current_velocity` topic along with the `/twist_cmd` topic to receive target linear and angular velocities. Additionally, this node subscribes to `/vehicle/dbw_enabled`, which indicates if the car is under dbw or driver control. This node publishes throttle, brake, and steering commands to the `/vehicle/throttle_cmd`, `/vehicle/brake_cmd`, and `/vehicle/steering_cmd` topics.



#### Developing environment construct

According to the introduction, this project can be done locally using a native Ubuntu System or Windows10 with provided linux VM or Workspace. The detailed setup instruction can be obtained from Udacity. If you have a fast computer, the comments from Student Hub recommend a native Ubuntu System is the best solution.

I have tried the **Windows10 with VM** on my old Laptop, it works fine with Camera off. But the car will run off  the  road with Camera on. 

I transfer to **Workspace**. It can work with Camera on. But with the Classifier I have trained using Google Colab, the latency is too much and will lead the car to run off the road.  For this Classifier, I have trained and  confirmed its ability to detect and classifier for the test images in the Colab with GPU. I have not tested it for the simulator for latency reason.

In order to finish the final project, in the Workspace Simulator, I used a very simple and low cost Classifier specific for this simulator. 



#### Traffic Light Classifier

##### Simple Classifier

The simple Classifier I have used is very cost saving and specific for the Simulator Camera Image with traditional computer vision technique. Followed is the basic pipeline for the image processing to detect the red circle in the camera image.

- get the camera image

- convert the image to HSV color space

- threshold the red color with in range function

- blur the image

- get the circle through HoughCircle

- judge whether the image has red light depending on the found circle from HoughCircle

  | ![](imgs\redlight.jpg) | ![](imgs\threshold.jpg) | ![](imgs\detected.jpg) |
  | ---------------------- | ----------------------- | ---------------------- |
|                        |                         |                        |
  
  

##### Complex Common Classifier

The simple Classifier will work because it is in the simulator and background are not complicated at all. For the classifier to work in real life, the model has to be more complex. The open source Google Tensorflow Object Detection API can be used here. The following is how I have done it with Google Colab. The pipeline can be found in the attached Jupyter Notebook file.  I have looked into this [tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10) online.



- Setup the Colab Jupyter Notebook for training 

  - Install the tensorflow and other depencies according to the tutorial
  - Download the tensorflow/models repository to google drive

  - COCO API installation
  - Protobuf Compilation
  - Add Libraries to PYTHONPATH
  - Testing the Installation
  - Choose the pre trained model in the Model Zoo （[faster_rcnn_resnet50_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz) ）

- Gather the training data

  I got the data from one of the post on line: [train data](https://drive.google.com/drive/folders/1kDV6NReRehsExP-mdMYqoYGszpVZhaEp), [test data](https://drive.google.com/drive/folders/1cg_Owcn-nXXpufHn2g6D7ElVE65ZWqag). If I have got enough time, I will gather data like the following:

  - gather camera image from simulator, udacity rosbag, Bosch TLD imag etc.
  - label the gathered image
  - transfer the labeled image to tf.record file
  - split the file to training and testing data

- Transfer Learning

  Train the model using the prepared data to change the model from a multi object classifier to a traffic light classifier.

- Output the trained model for later use

  The model was saved as forzen_graph and can be used in anywhere.

  | ![](imgs\red.jpg) | ![](imgs\yellow.jpg) | ![](imgs\green.jpg) |
| ----------------- | -------------------- | ------------------- |
  |                   |                      |                     |

  Finally, I have tested the model in the workspace, but because of latency problem , the model was not tested in the complete loop in the simulator.
  
  ![](imgs\test.gif)



### Result  and discussion

Finally I tested the whole system using the simple classifier on the simulator in the udacity workspace. The car can drive along the set way points and drive according to the traffic light. The car will stop before a red light and will drive with the yellow or green light. The followed gif can show the result.

------

------

## Followed are the original Instruction from Udacity

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the "uWebSocketIO Starter Guide" found in the classroom (see Extended Kalman Filter Project lesson).

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

### Other library/driver information
Outside of `requirements.txt`, here is information on other driver/library versions used in the simulator and Carla:

Specific to these libraries, the simulator grader and Carla use the following:

|        | Simulator | Carla  |
| :-----------: |:-------------:| :-----:|
| Nvidia driver | 384.130 | 384.130 |
| CUDA | 8.0.61 | 8.0.61 |
| cuDNN | 6.0.21 | 6.0.21 |
| TensorRT | N/A | N/A |
| OpenCV | 3.2.0-dev | 2.4.8 |
| OpenMP | N/A | N/A |

We are working on a fix to line up the OpenCV versions between the two.
