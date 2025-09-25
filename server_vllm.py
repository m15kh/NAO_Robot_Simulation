# Copyright 1996-2024 Cyberbotics Ltd.
# Licensed under the Apache License, Version 2.0
"""
Nao controller with WebSocket control + external inference server.
- Local WS server: ws://0.0.0.0:8765
- External inference server: url_server
"""

from controller import Robot, Keyboard, Motion

import asyncio
import base64
import json
import queue
import re
import threading
import time

import cv2
import numpy as np
import websockets  # pip install websockets
import logging
from colorama import Fore, Style, init


HEX_RGB = re.compile(r"^#?[0-9A-Fa-f]{6}$")

# Initialize colorama for colored output
init(autoreset=True)

# Clear the log file at the start of the program
with open("server.log", "w"):
    pass

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("server.log", mode="w"),
        logging.StreamHandler()
    ]
)

# Ensure full log messages are displayed
logging.getLogger().handlers[0].setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().handlers[1].setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Helper function for colored logs
def log_info(message):
    logging.info(Fore.GREEN + message)

def log_warning(message):
    logging.warning(Fore.YELLOW + message)

def log_error(message):
    logging.error(Fore.RED + message)

def log_debug(message):
    logging.debug(Fore.CYAN + message)

class Nao(Robot):
    PHALANX_MAX = 8

    # -------------------- WS HELPERS --------------------
    def _ws_pack_telemetry(self):
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

    def _frame_to_jpeg(self, camera):
        w, h = camera.getWidth(), camera.getHeight()
        raw_bgra = camera.getImage()
        arr = np.frombuffer(raw_bgra, dtype=np.uint8).reshape((h, w, 4))
        bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return None
        return base64.b64encode(enc).decode("ascii")

    async def _ws_handler(self, websocket):
        log_info("[WS] Client connected")
        await websocket.send(json.dumps({"type": "hello", "robot": "nao", "msg": "connected"}))
        try:
            while True:
                try:
                    while True:
                        out = self.telemetry_queue.get_nowait()
                        await websocket.send(json.dumps(out))
                except queue.Empty:
                    pass

                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=0.02)
                    data = json.loads(msg)
                    self.command_queue.put(data)
                except asyncio.TimeoutError:
                    log_debug("[WS] No message received (timeout)")
                    pass

        except websockets.exceptions.ConnectionClosed:
            log_warning("[WS] Client disconnected")

    async def _ws_server(self, host="0.0.0.0", port=8765):
        log_info(f"[WS] Starting WebSocket server on ws://{host}:{port}")
        async with websockets.serve(self._ws_handler, host, port, ping_interval=20, ping_timeout=20):
            log_info(f"[WS] Local WebSocket server running on ws://{host}:{port}")
            await asyncio.Future()

    def _ws_thread_target(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._ws_server())
        loop.close()

    def _start_ws_server(self):
        if getattr(self, "_ws_thread", None):
            return
        self.command_queue = queue.Queue()
        self.telemetry_queue = queue.Queue()
        self._ws_thread = threading.Thread(target=self._ws_thread_target, daemon=True)
        self._ws_thread.start()
        print("[WS] Local server thread started.")

    # -------------------- EXTERNAL SERVER --------------------
    async def _ext_client(self):
        log_info(f"[EXT] Connecting to external server: {self.ext_uri}")
        async with websockets.connect(self.ext_uri) as ws:
            log_info("[EXT] Connected to external server")
            while True:
                b64 = self._frame_to_jpeg(self.cameraTop)
                if not b64:
                    await asyncio.sleep(0.5)
                    continue

                await ws.send(json.dumps({"type": "frame", "image": b64}))

                try:
                    resp = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    data = json.loads(resp)
                    action = data.get("response") #DEBUG revive data from external server
                    if action:
                        print("[EXT] Got action:", action)
                        self._apply_ext_action(action)
                except asyncio.TimeoutError:
                    log_warning("[EXT] No response from external server (timeout)")

                await asyncio.sleep(0.5)  # 2 FPS

    def _apply_ext_action(self, action):
        log_debug(f"[EXT] Applying action: {action}")
        action = action.lower()
        if action == "move_forward":  # Fixed case-insensitive comparison
            self.startMotion(self.forwards)
        elif action == "turn_left":
            self.startMotion(self.turnLeft40)
        elif action == "turn_right":
            self.startMotion(self.turnRight40)
        elif action == "human_not_detected":  # TODO: if no human, turn around itself!
            pass
            # self.startMotion(None)  # BUG: maybe approach

    def _start_ext_client(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._ext_client())
        loop.close()

    # -------------------- DEMO FUNCTIONS --------------------
    def loadMotionFiles(self):
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
        log_debug(f"[MOTION] Starting motion: {motion}")
        if self.currentlyPlaying:
            self.currentlyPlaying.stop()
        if motion:
            motion.play()
            self.currentlyPlaying = motion
        else:
            self.currentlyPlaying = None

    def printHelp(self):
        log_info("[HELP] Displaying help information")
        print('----------nao_ws_demo----------')
        print('Keyboard works as before; plus WS on ws://<host>:8765')
        print('JSON:')
        print('  {"type":"cmd","action":"forward"}')
        print('  {"type":"cmd","action":"led","rgb":"#00FF00"}')
        print('  {"type":"cmd","action":"hand","state":"open"}')
        print('  {"type":"get","sensor":"imu|gps|us|fsr|frame","camera":"top|bottom"}')

    def findAndEnableDevices(self):
        self.timeStep = int(self.getBasicTimeStep())

        self.cameraTop = self.getDevice("CameraTop")
        self.cameraBottom = self.getDevice("CameraBottom")
        self.cameraTop.enable(4 * self.timeStep)
        self.cameraBottom.enable(4 * self.timeStep)

        self.accelerometer = self.getDevice('accelerometer'); self.accelerometer.enable(4 * self.timeStep)
        self.gyro = self.getDevice('gyro'); self.gyro.enable(4 * self.timeStep)
        self.gps = self.getDevice('gps'); self.gps.enable(4 * self.timeStep)
        self.inertialUnit = self.getDevice('inertial unit'); self.inertialUnit.enable(self.timeStep)

        self.us = []
        for name in ['Sonar/Left', 'Sonar/Right']:
            dev = self.getDevice(name); dev.enable(self.timeStep); self.us.append(dev)

        self.fsr = []
        for name in ['LFsr', 'RFsr']:
            dev = self.getDevice(name); dev.enable(self.timeStep); self.fsr.append(dev)

        self.lfootlbumper = self.getDevice('LFoot/Bumper/Left')
        self.lfootrbumper = self.getDevice('LFoot/Bumper/Right')
        self.rfootlbumper = self.getDevice('RFoot/Bumper/Left')
        self.rfootrbumper = self.getDevice('RFoot/Bumper/Right')
        for b in [self.lfootlbumper, self.lfootrbumper, self.rfootrbumper, self.rfootlbumper]:
            b.enable(self.timeStep)

        self.leds = [
            self.getDevice('ChestBoard/Led'),
            self.getDevice('RFoot/Led'),
            self.getDevice('LFoot/Led'),
            self.getDevice('Face/Led/Right'),
            self.getDevice('Face/Led/Left'),
            self.getDevice('Ears/Led/Right'),
            self.getDevice('Ears/Led/Left'),
        ]

        self.lphalanx, self.rphalanx = [], []
        self.maxPhalanxMotorPosition, self.minPhalanxMotorPosition = [], []
        for i in range(0, self.PHALANX_MAX):
            lp = self.getDevice(f"LPhalanx{i+1}")
            rp = self.getDevice(f"RPhalanx{i+1}")
            self.lphalanx.append(lp); self.rphalanx.append(rp)
            self.maxPhalanxMotorPosition.append(rp.getMaxPosition())
            self.minPhalanxMotorPosition.append(rp.getMinPosition())

        self.RShoulderPitch = self.getDevice("RShoulderPitch")
        self.LShoulderPitch = self.getDevice("LShoulderPitch")

        self.keyboard = self.getKeyboard(); self.keyboard.enable(10 * self.timeStep)

    def __init__(self):
        log_info("[INIT] Initializing Nao robot")
        Robot.__init__(self)
        self.currentlyPlaying = False
        self.jpeg_quality = 70
        self.findAndEnableDevices()
        self.loadMotionFiles()
        self.printHelp()

        # start local + external servers
        self._start_ws_server()
        self.ext_uri = "url_server"
        self._ext_thread = threading.Thread(target=self._start_ext_client, daemon=True)
        self._ext_thread.start()
        log_info("[INIT] Initialization complete")

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

        elif tp == "get":
            sensor = (cmd.get("sensor") or "").lower()
            if sensor in ("imu", "gps", "us", "fsr"):
                self.telemetry_queue.put(self._ws_pack_telemetry())
            elif sensor == "frame":
                which = (cmd.get("camera") or "top").lower()
                if which == "bottom":
                    self.telemetry_queue.put(self._frame_to_jpeg(self.cameraBottom))
                else:
                    self.telemetry_queue.put(self._frame_to_jpeg(self.cameraTop))

    # -------------------- MAIN LOOP --------------------
    def run(self):
        log_info("[RUN] Starting main loop")
        self.handWave.setLoop(True)
        self.handWave.play()
        self.currentlyPlaying = self.handWave

        while self.step(self.timeStep) != -1:
            if self.keyboard.getKey() > 0:
                break
            try:
                cmd = self.command_queue.get_nowait()
                self.command_queue.put(cmd)
                break
            except queue.Empty:
                pass

        self.handWave.setLoop(False)

        while self.step(self.timeStep) != -1:
            try:
                while True:
                    cmd = self.command_queue.get_nowait()
                    self._apply_ws_command(cmd)
            except queue.Empty:
                pass

            key = self.keyboard.getKey()
            if key == Keyboard.LEFT:  self.startMotion(self.sideStepLeft)
            elif key == Keyboard.RIGHT: self.startMotion(self.sideStepRight)
            elif key == Keyboard.UP:    self.startMotion(self.forwards)
            elif key == Keyboard.DOWN:  self.startMotion(self.backwards)
            elif key == Keyboard.LEFT | Keyboard.SHIFT:  self.startMotion(self.turnLeft40)
            elif key == Keyboard.RIGHT | Keyboard.SHIFT: self.startMotion(self.turnRight40)
            elif key == ord('H'): self.printHelp()

        log_info("[RUN] Main loop terminated")


# create instance and run
robot = Nao()
robot.run()
