# Copyright 1996-2024 Cyberbotics Ltd.
# Licensed under the Apache License, Version 2.0
"""
Nao controller with WebSocket control + telemetry + camera frames (JPEG if OpenCV available, else BGRA).
- WS server: ws://0.0.0.0:8765
- JSON protocol (examples):
  {"type":"cmd","action":"forward|backward|left|right|turn_left|turn_right"}
  {"type":"cmd","action":"hand","state":"open|close"}
  {"type":"cmd","action":"led","rgb":"#RRGGBB"}
  {"type":"get","sensor":"imu|gps|us|fsr|frame","camera":"top|bottom"}  # frame is on-demand
  # Periodic telemetry is also pushed automatically each loop
"""

from controller import Robot, Keyboard, Motion

import asyncio
import base64
import json
import queue
import re
import threading
import time
# import SmartAiTool
# Optional OpenCV (for JPEG compression)
# try:
#     import cv2
#     import numpy as np
#     CV2_OK = True
# except Exception:
#     CV2_OK = False

import cv2
import numpy as np
from ultralytics import YOLO  # Import YOLO from Ultralytics

# websockets
import websockets  # pip install websockets


HEX_RGB = re.compile(r"^#?[0-9A-Fa-f]{6}$")


class Nao(Robot):
    PHALANX_MAX = 8

     # -------------------- WS HELPERS --------------------
    def _ws_pack_telemetry(self): #@m15kh automatically sent this feature to the client i deactivate this feature!
        rpy = self.inertialUnit.getRollPitchYaw()
        gps = self.gps.getValues()
        usL = self.us[0].getValue()
        usR = self.us[1].getValue()
        return {
            "type": "telemetry",
            "imu": {"roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2]},
            "gps": {"x": gps[0], "y": gps[1], "z": gps[2]},
            "ultrasound": {"left": usL, "right": usR},
            "ts": time.time(),
        }

    def _frame_to_message(self, camera, cam_name="top"):
        """Return a dict with either JPEG(base64) or raw BGRA(base64), and YOLO bounding boxes."""
        w, h = camera.getWidth(), camera.getHeight()
        raw_bgra = camera.getImage()  # bytes, len = w*h*4 (BGRA)

        # Convert BGRA bytes -> numpy -> BGR
        arr = np.frombuffer(raw_bgra, dtype=np.uint8).reshape((h, w, 4))
        bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

        # Run YOLO inference
        classes = [
            # humans (multiple synonyms help recall)
            "person", "human", "man", "woman",
            # broad catch-alls
            "object", "thing",
            # common categories (extend as needed)
            "car", "dog", "cat", "bottle", "cup", "chair", "table", "laptop",
            "phone", "backpack", "bicycle", "motorcycle", "traffic light",
        ]
        
        # Define class indices for the desired classes
        class_indices = [0, 1, 2, 3, 5, 7, 15, 16, 17, 39, 41, 56, 57, 60, 62, 63, 64, 67]  # Replace with actual indices

        results = self.yolo_model.predict(source=bgr, conf=0.5, classes=class_indices, verbose=False)  # Use class indices
        bounding_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = float(box.conf[0])  # Confidence score
                bounding_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf})

        # Evaluate human position in the frame
        human_positions = self._evaluate_human_position(bounding_boxes, w)
        print("w:",w)

        # Encode frame as JPEG
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            # Fallback to raw BGRA if JPEG encoding fails
            b64 = base64.b64encode(raw_bgra).decode("ascii")
            return {"type": "frame", "camera": cam_name, "format": "BGRA",
                    "width": w, "height": h, "data": b64, "detections": bounding_boxes, "human_positions": human_positions}

        b64 = base64.b64encode(enc).decode("ascii")
        return {"type": "frame", "camera": cam_name, "format": "JPEG",
                "width": w, "height": h, "data": b64, "detections": bounding_boxes, "human_positions": human_positions}
        # else: #@m15kh
        #     # No OpenCV: send raw BGRA
        #     b64 = base64.b64encode(raw_bgra).decode("ascii")
        #     return {"type": "frame", "camera": cam_name, "format": "BGRA",
        #             "width": w, "height": h, "data": b64}

    async def _ws_handler(self, websocket):
        await websocket.send(json.dumps({"type": "hello", "robot": "nao", "msg": "connected"})) #LOG hello to client for sent connection
        try:
            while True:
                # Flush any pending outgoing messages first
                try:
                    while True: #LOG #always runs until queue is empty!
                        out = self.telemetry_queue.get_nowait() #same as simple .get() instead has expection too for managment
                        await websocket.send(json.dumps(out))
                except queue.Empty:
                    pass

                # Poll incoming with small timeout
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=0.02)
                    data = json.loads(msg)
                    self.command_queue.put(data) #add command to robots!
                except asyncio.TimeoutError:
                    pass
                
        except websockets.exceptions.ConnectionClosed:
            return

    async def _ws_server(self, host="0.0.0.0", port=8765):
        async with websockets.serve(self._ws_handler, host, port, ping_interval=20, ping_timeout=20):
            print(f"[WS] WebSocket server listening on ws://{host}:{port}")
            await asyncio.Future() #They use a Future because it’s an easy way to tell Python: don’t exit, just stay alive until the program is stopped

    def _ws_thread_target(self):
        if websockets is None:
            print("[WS] websockets not available; skipping server.")
            return
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_server())
        finally:
            loop.close()

    def _start_ws_server(self):
        if getattr(self, "_ws_thread", None):
            return
        self.command_queue = queue.Queue()
        self.telemetry_queue = queue.Queue() # #
        # self.images_queue = queue.Queue() #@m15kh added
        self._ws_thread = threading.Thread(target=self._ws_thread_target, daemon=True)
        self._ws_thread.start()
        print("[WS] Server thread started.")

    # -------------------- DEMO FUNCTIONS --------------------
    def loadMotionFiles(self):
        # Adjust paths depending on your controller folder layout
        self.handWave     = Motion('motions/HandWave.motion')
        self.forwards     = Motion('motions/Forwards50.motion')
        self.backwards    = Motion('motions/Backwards.motion')
        self.sideStepLeft = Motion('motions/SideStepLeft.motion')
        self.sideStepRight= Motion('motions/SideStepRight.motion')
        self.turnLeft40   = Motion('motions/TurnLeft10.motion')
        self.turnRight40  = Motion('motions/TurnRight10.motion')
        self.taiChi       = Motion('motions/TaiChi.motion')
        self.wipeForhead  = Motion('motions/WipeForehead.motion')

    def startMotion(self, motion):
        if self.currentlyPlaying:
            self.currentlyPlaying.stop()
        if motion:  # Only play the motion if it's not None
            motion.play()
            self.currentlyPlaying = motion
        else:
            self.currentlyPlaying = None

    def printAcceleration(self):
        acc = self.accelerometer.getValues()
        print('----------accelerometer----------')
        print('acceleration: [ x y z ] = [%f %f %f]' % (acc[0], acc[1], acc[2]))

    def printGyro(self):
        vel = self.gyro.getValues()
        print('----------gyro----------')
        print('angular velocity: [ x y ] = [%f %f]' % (vel[0], vel[1]))

    def printGps(self):
        p = self.gps.getValues()
        print('----------gps----------')
        print('position: [ x y z ] = [%f %f %f]' % (p[0], p[1], p[2]))

    def printInertialUnit(self):
        rpy = self.inertialUnit.getRollPitchYaw()
        print('----------inertial unit----------')
        print('roll/pitch/yaw: [%f %f %f]' % (rpy[0], rpy[1], rpy[2]))

    def printFootSensors(self):
        fsv = [self.fsr[0].getValues(), self.fsr[1].getValues()]
        left, right = [], []
        newtonsLeft = newtonsRight = 0
        left.append(fsv[0][2] / 3.4 + 1.5 * fsv[0][0] + 1.15 * fsv[0][1])
        left.append(fsv[0][2] / 3.4 + 1.5 * fsv[0][0] - 1.15 * fsv[0][1])
        left.append(fsv[0][2] / 3.4 - 1.5 * fsv[0][0] - 1.15 * fsv[0][1])
        left.append(fsv[0][2] / 3.4 - 1.5 * fsv[0][0] + 1.15 * fsv[0][1])
        right.append(fsv[1][2] / 3.4 + 1.5 * fsv[1][0] + 1.15 * fsv[1][1])
        right.append(fsv[1][2] / 3.4 + 1.5 * fsv[1][0] - 1.15 * fsv[1][1])
        right.append(fsv[1][2] / 3.4 - 1.5 * fsv[1][0] - 1.15 * fsv[1][1])
        right.append(fsv[1][2] / 3.4 - 1.5 * fsv[1][0] + 1.15 * fsv[1][1])
        for i in range(len(left)):
            left[i]  = max(min(left[i], 25), 0)
            right[i] = max(min(right[i], 25), 0)
            newtonsLeft  += left[i]
            newtonsRight += right[i]
        print('----------foot sensors----------')
        print('+ left ---- right +')
        print('+-------+ +-------+')
        print(f'|{round(left[0],1)}  {round(left[1],1)}| |{round(right[0],1)}  {round(right[1],1)}|  front')
        print('| ----- | | ----- |')
        print(f'|{round(left[3],1)}  {round(left[2],1)}| |{round(right[3],1)}  {round(right[2],1)}|  back')
        print('+-------+ +-------+')
        print('total: %f Newtons, %f kilograms' %
              ((newtonsLeft + newtonsRight), ((newtonsLeft + newtonsRight) / 9.81)))

    def printFootBumpers(self):
        ll = self.lfootlbumper.getValue()
        lr = self.lfootrbumper.getValue()
        rl = self.rfootlbumper.getValue()
        rr = self.rfootrbumper.getValue()
        print('----------foot bumpers----------')
        print('+ left ------ right +')
        print('+--------+ +--------+')
        print(f'|{ll}  {lr}| |{rl}  {rr}|')
        print('|        | |        |')
        print('|        | |        |')
        print('+--------+ +--------+')

    def printUltrasoundSensors(self):
        dist = [self.us[i].getValue() for i in range(len(self.us))]
        print('-----ultrasound sensors-----')
        print('left: %f m, right %f m' % (dist[0], dist[1]))

    def printCameraImage(self, camera):
        scaled = 2
        width = camera.getWidth()
        height = camera.getHeight()
        image = camera.getImage()
        print('----------camera image (gray levels)---------')
        print('original resolution: %d x %d, scaled to %d x %f'
              % (width, height, width / scaled, height / scaled))
        for y in range(0, height // scaled):
            line = ''
            for x in range(0, width // scaled):
                gray = camera.imageGetGray(image, width, x * scaled, y * scaled) * 9 / 255
                line += str(int(gray))
            print(line)

    def setAllLedsColor(self, rgb):
        for i in range(0, len(self.leds)):
            self.leds[i].set(rgb)
        self.leds[5].set(rgb & 0xFF)
        self.leds[6].set(rgb & 0xFF)

    def setHandsAngle(self, angle):
        for i in range(0, self.PHALANX_MAX):
            clampedAngle = max(min(angle, self.maxPhalanxMotorPosition[i]), self.minPhalanxMotorPosition[i])
            if len(self.rphalanx) > i and self.rphalanx[i] is not None:
                self.rphalanx[i].setPosition(clampedAngle)
            if len(self.lphalanx) > i and self.lphalanx[i] is not None:
                self.lphalanx[i].setPosition(clampedAngle)

    def printHelp(self):
        print('----------nao_ws_demo----------')
        print('Keyboard works as before; plus WS on ws://<host>:8765')
        print('JSON:')
        print('  {"type":"cmd","action":"forward"}')
        print('  {"type":"cmd","action":"led","rgb":"#00FF00"}')
        print('  {"type":"cmd","action":"hand","state":"open"}')
        print('  {"type":"get","sensor":"imu|gps|us|fsr|frame","camera":"top|bottom"}')

    def findAndEnableDevices(self):
        self.timeStep = int(self.getBasicTimeStep())

        # cameras
        self.cameraTop = self.getDevice("CameraTop")
        self.cameraBottom = self.getDevice("CameraBottom")
        self.cameraTop.enable(4 * self.timeStep)
        self.cameraBottom.enable(4 * self.timeStep)

        # sensors
        self.accelerometer = self.getDevice('accelerometer'); self.accelerometer.enable(4 * self.timeStep)
        self.gyro = self.getDevice('gyro'); self.gyro.enable(4 * self.timeStep)
        self.gps = self.getDevice('gps'); self.gps.enable(4 * self.timeStep)
        self.inertialUnit = self.getDevice('inertial unit'); self.inertialUnit.enable(self.timeStep)

        # ultrasound
        self.us = []
        for name in ['Sonar/Left', 'Sonar/Right']:
            dev = self.getDevice(name); dev.enable(self.timeStep); self.us.append(dev)

        # foot sensors
        self.fsr = []
        for name in ['LFsr', 'RFsr']:
            dev = self.getDevice(name); dev.enable(self.timeStep); self.fsr.append(dev)

        # bumpers
        self.lfootlbumper = self.getDevice('LFoot/Bumper/Left')
        self.lfootrbumper = self.getDevice('LFoot/Bumper/Right')
        self.rfootlbumper = self.getDevice('RFoot/Bumper/Left')
        self.rfootrbumper = self.getDevice('RFoot/Bumper/Right')
        for b in [self.lfootlbumper, self.lfootrbumper, self.rfootrbumper, self.rfootlbumper]:
            b.enable(self.timeStep)

        # leds
        self.leds = [
            self.getDevice('ChestBoard/Led'),
            self.getDevice('RFoot/Led'),
            self.getDevice('LFoot/Led'),
            self.getDevice('Face/Led/Right'),
            self.getDevice('Face/Led/Left'),
            self.getDevice('Ears/Led/Right'),
            self.getDevice('Ears/Led/Left'),
        ]

        # hands
        self.lphalanx, self.rphalanx = [], []
        self.maxPhalanxMotorPosition, self.minPhalanxMotorPosition = [], []
        for i in range(0, self.PHALANX_MAX):
            lp = self.getDevice(f"LPhalanx{i+1}")
            rp = self.getDevice(f"RPhalanx{i+1}")
            self.lphalanx.append(lp); self.rphalanx.append(rp)
            self.maxPhalanxMotorPosition.append(rp.getMaxPosition())
            self.minPhalanxMotorPosition.append(rp.getMinPosition())

        # shoulders
        self.RShoulderPitch = self.getDevice("RShoulderPitch")
        self.LShoulderPitch = self.getDevice("LShoulderPitch")

        # keyboard
        self.keyboard = self.getKeyboard(); self.keyboard.enable(10 * self.timeStep)

    def __init__(self):
        Robot.__init__(self)
        self.currentlyPlaying = False
        self.jpeg_quality = 70  # JPEG quality (0-100) if OpenCV available
        self.findAndEnableDevices()
        self.loadMotionFiles()
        self.printHelp()

        # Load YOLO model (e.g., pretrained on COCO dataset)
        self.yolo_model = YOLO("weights/yolov8n.pt")  # Use a lightweight model for performance

        # periodic frame control (if you want push-frames every N ticks)
        self._frame_tick = 0
        self._frame_every = 10  # 0 = disabled; set e.g. 5 to push each 5 steps

        self._start_ws_server()  # Start WebSocket server

    # -------------------- CMD APPLY --------------------
    def _apply_ws_command(self, cmd):
        if not isinstance(cmd, dict):
            return
        tp = cmd.get("type")

        if tp == "cmd":
            action = (cmd.get("action") or "").lower()
            if action == "forward":
                self.startMotion(self.forwards)
            elif action == "backward":
                self.startMotion(self.backwards)
            elif action == "left":
                self.startMotion(self.sideStepLeft)
            elif action == "right":
                self.startMotion(self.sideStepRight)
            elif action == "turn_left":
                self.startMotion(self.turnLeft40)
            elif action == "turn_right":
                self.startMotion(self.turnRight40)
            elif action == "led":
                rgb = cmd.get("rgb", "#000000")
                if isinstance(rgb, str) and HEX_RGB.match(rgb):
                    rgb = rgb.lstrip("#")
                    self.setAllLedsColor(int(rgb, 16))
            elif action == "hand":
                st = (cmd.get("state") or "").lower()
                if st == "open":
                    self.setHandsAngle(0.0)
                elif st == "close":
                    self.setHandsAngle(0.96)

        elif tp == "get": #NOTE when receive data from client
            sensor = (cmd.get("sensor") or "").lower()
            if sensor in ("imu", "gps", "us", "fsr"):
                self.telemetry_queue.put(self._ws_pack_telemetry())
            elif sensor == "frame":
                which = (cmd.get("camera") or "top").lower()
                if which == "bottom":
                    self.telemetry_queue.put(self._frame_to_message(self.cameraBottom, "bottom"))
                else:
                    self.telemetry_queue.put(self._frame_to_message(self.cameraTop, "top"))

    # -------------------- MAIN LOOP --------------------
    def run(self):
        # Start with a hand wave until the first interaction
        self.handWave.setLoop(True)
        self.handWave.play()
        self.currentlyPlaying = self.handWave

        while self.step(self.timeStep) != -1:
            if self.keyboard.getKey() > 0:
                break
            try:
                cmd = self.command_queue.get_nowait()
                self.command_queue.put(cmd)  # Put back, main loop will handle
                break
            except queue.Empty:
                pass

        self.handWave.setLoop(False)

        while self.step(self.timeStep) != -1:
            # Get the camera frame and evaluate human position
            frame_message = self._frame_to_message(self.cameraTop, "top")
            if frame_message:
                human_positions = frame_message["human_positions"]

                # Check if a human is detected
                if any(human_positions.values()):
                    if human_positions["left"]:
                        self.startMotion(self.turnLeft40)  # Turn left to center the human
                    elif human_positions["right"]:
                        self.startMotion(self.turnRight40)  # Turn right to center the human
                    elif human_positions["center"]:
                        distance_left = self.us[0].getValue()
                        distance_right = self.us[1].getValue()
                        min_distance = min(distance_left, distance_right)

                        if min_distance > 0.5:  # Move forward if the human is not too close
                            self.startMotion(self.forwards)
                        else:
                            print("Human reached. Stopping.")
                            self.startMotion(None)  # Stop the robot
                else:
                    print("No human detected. Waiting...")
                    self.startMotion(None)  # Stop the robot if no human is detected

            # Handle WebSocket commands
            try:
                while True:
                    cmd = self.command_queue.get_nowait()
                    self._apply_ws_command(cmd)
            except queue.Empty:
                pass

            # keyboard as before
            key = self.keyboard.getKey()
            if key == Keyboard.LEFT:  self.startMotion(self.sideStepLeft)
            elif key == Keyboard.RIGHT: self.startMotion(self.sideStepRight)
            elif key == Keyboard.UP:    self.startMotion(self.forwards)
            elif key == Keyboard.DOWN:  self.startMotion(self.backwards)
            elif key == Keyboard.LEFT | Keyboard.SHIFT:  self.startMotion(self.turnLeft40)
            elif key == Keyboard.RIGHT | Keyboard.SHIFT: self.startMotion(self.turnRight40)
            elif key == ord('A'): self.printAcceleration()
            elif key == ord('G'): self.printGyro()
            elif key == ord('S'): self.printGps()
            elif key == ord('I'): self.printInertialUnit()
            elif key == ord('F'): self.printFootSensors()
            elif key == ord('B'): self.printFootBumpers()
            elif key == ord('U'): self.printUltrasoundSensors()
            elif key == ord('T'): self.startMotion(self.taiChi)
            elif key == ord('W'): self.startMotion(self.wipeForhead)
            elif key == Keyboard.HOME: self.printCameraImage(self.cameraTop)
            elif key == Keyboard.END:  self.printCameraImage(self.cameraBottom)
            elif key == Keyboard.PAGEUP:   self.setHandsAngle(0.96)
            elif key == Keyboard.PAGEDOWN: self.setHandsAngle(0.0)
            elif key == ord('7'): self.setAllLedsColor(0xff0000)
            elif key == ord('8'): self.setAllLedsColor(0x00ff00)
            elif key == ord('9'): self.setAllLedsColor(0x0000ff)
            elif key == ord('0'): self.setAllLedsColor(0x000000)
            elif key == ord('H'): self.printHelp()

            # periodic telemetry push
            # self.telemetry_queue.put(self._ws_pack_telemetry())#sent autoamcllt to server! #@m15kh

            # optional periodic frame push (disabled by default) #NOTE automatcllt sent to client
            if self._frame_every :
                self._frame_tick += 1
                if self._frame_tick % self._frame_every == 0:
                    # self.telemetry_queue.put(self._frame_to_message(self.cameraTop, "top"))
                    frame_message = self._frame_to_message(self.cameraTop, "top")
                    print("@@@im here@@@@@")
                    if frame_message:  # Only send if a message is returned
                        self.telemetry_queue.put(frame_message)

    def _evaluate_human_position(self, bounding_boxes, frame_width):
        """
        Evaluate which part of the frame (left, center, right) contains a human.
        :param bounding_boxes: List of bounding boxes with coordinates and confidence.
        :param frame_width: Width of the frame.
        :return: A dictionary indicating human presence in left, center, and right.
        """
        # Define the boundaries for the three regions
        left_boundary = frame_width // 3
        right_boundary = 2 * frame_width // 3

        regions = {"left": False, "center": False, "right": False}

        for box in bounding_boxes:
            x1, x2 = box["x1"], box["x2"]

            # Check if the bounding box overlaps with the left region
            if x1 < left_boundary:
                regions["left"] = True
            # Check if the bounding box overlaps with the center region
            if x1 < right_boundary and x2 > left_boundary:
                regions["center"] = True
            # Check if the bounding box overlaps with the right region
            if x2 > right_boundary:
                regions["right"] = True

        return regions
# create instance and run
robot = Nao()
robot.run()
