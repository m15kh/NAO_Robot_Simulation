# Human Detection and NAO Robot Motion Control Project

This project is designed to implement an intelligent system for **human detection** and **motion control of the NAO robot** automatically. The system uses a **webcam** and the **YOLOv8n** model to detect human positions in the image and then issues appropriate movement commands for the robot. The **NAO robot** is simulated in the **Webots** environment, and the movement commands are received through **WebSocket**.

## Project Sections

### 1. Human Detection with Webcam

* A **regular webcam** is used to capture live images.
* The **YOLOv8n model** is loaded to detect humans in the image.
* For each frame, humans are detected, and their position relative to the robot is assessed (left, right, or center of the image).
* **Bounding box height** is used to estimate the distance of the human from the robot.

### 2. Executing Movement Commands on NAO Robot with WebSocket

* After detecting a human, the system issues movement commands like **turn left, turn right, move forward**, or **stop**.
* Commands are sent to the robot via **WebSocket**. WebSocket serves as a bi-directional protocol for sending and receiving data between the robot and external software.
* The robot receives and executes movement commands in real-time.

## Demo

Check out our demo video to see WeBot in action:

[![WeBot Demo](assets/demo.bmp)](assets/demo.mp4)



## Features

* **Human detection in the image** using the YOLO model.
* **Real-time communication** with the robot via WebSocket.
* **Robot motion control** based on the detected human position.

## Setup

1. Ensure the system detects and activates the webcam correctly.
2. Load the **YOLOv8n model** and integrate it into the project.
3. Set up **WebSocket** to send commands to the robot.
4. Run the project in the **Webots** simulator and control the robot's movement in response to the detected human.

