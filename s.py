# Copyright 1996-2024 Cyberbotics Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
"""Nao controller with WebSocket control + telemetry."""

from controller import Robot, Keyboard, Motion

# === New imports for WebSocket/threads/queues ===
import asyncio
import json
import threading
import queue
import re

try:
    import websockets  # pip install websockets
except ImportError:
    websockets = None
    print("[WARN] 'websockets' not installed. Run: pip install websockets")

HEX_RGB = re.compile(r"^#?[0-9A-Fa-f]{6}$")


class Nao(Robot):
    PHALANX_MAX = 8

    # ------------- WEBSOCKET BRIDGE (NEW) -------------
    def _ws_pack_telemetry(self):
        """Collect a light telemetry snapshot as dict."""
        t = {}
        # inertial
        rpy = self.inertialUnit.getRollPitchYaw()
        t["imu"] = {"roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2]}
        # gps
        p = self.gps.getValues()
        t["gps"] = {"x": p[0], "y": p[1], "z": p[2]}
        # ultrasound
        dist = [us.getValue() for us in self.us]
        t["ultrasound"] = {"left": dist[0], "right": dist[1]}
        return {"type": "telemetry", **t}

    async def _ws_handler(self, websocket):
        """Async handler per client."""
        # Simple welcome
        await websocket.send(json.dumps({"type": "hello", "robot": "nao", "msg": "connected"}))
        # Consumer/producer loop
        try:
            while True:
                # 1) drain outgoing telemetry first (non-blocking)
                try:
                    while True:
                        out = self.telemetry_queue.get_nowait()
                        await websocket.send(json.dumps(out))
                except queue.Empty:
                    pass

                # 2) poll incoming with small timeout
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=0.02)
                    data = json.loads(msg)
                    # push to command queue
                    self.command_queue.put(data)
                except asyncio.TimeoutError:
                    pass
        except websockets.exceptions.ConnectionClosed:
            return

    async def _ws_server(self, host="0.0.0.0", port=8765):
        async with websockets.serve(self._ws_handler, host, port, ping_interval=20, ping_timeout=20):
            print(f"[WS] WebSocket server listening on ws://{host}:{port}")
            # Keep server alive forever
            await asyncio.Future()

    def _ws_thread_target(self):
        """Run asyncio server in a dedicated thread."""
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
        """Start the WS server thread once."""
        if getattr(self, "_ws_thread", None):
            return
        self.command_queue = queue.Queue()
        self.telemetry_queue = queue.Queue()
        self._ws_thread = threading.Thread(target=self._ws_thread_target, daemon=True)
        self._ws_thread.start()
        print("[WS] Server thread started.")

    # ------------- ORIGINAL DEMO FUNCTIONS -------------
    def loadMotionFiles(self):
        self.handWave = Motion('motions/HandWave.motion')
        self.forwards = Motion('motions/Forwards50.motion')
        self.backwards = Motion('motions/Backwards.motion')
        self.sideStepLeft = Motion('motions/SideStepLeft.motion')
        self.sideStepRight = Motion('motions/SideStepRight.motion')
        self.turnLeft60 = Motion('motions/TurnLeft60.motion')
        self.turnRight60 = Motion('motions/TurnRight60.motion')
        self.taiChi = Motion('motions/TaiChi.motion')
        self.wipeForhead = Motion('motions/WipeForehead.motion')

    def startMotion(self, motion):
        if self.currentlyPlaying:
            self.currentlyPlaying.stop()
        motion.play()
        self.currentlyPlaying = motion

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
        newtonsLeft = 0
        newtonsRight = 0
        left.append(fsv[0][2] / 3.4 + 1.5 * fsv[0][0] + 1.15 * fsv[0][1])
        left.append(fsv[0][2] / 3.4 + 1.5 * fsv[0][0] - 1.15 * fsv[0][1])
        left.append(fsv[0][2] / 3.4 - 1.5 * fsv[0][0] - 1.15 * fsv[0][1])
        left.append(fsv[0][2] / 3.4 - 1.5 * fsv[0][0] + 1.15 * fsv[0][1])

        right.append(fsv[1][2] / 3.4 + 1.5 * fsv[1][0] + 1.15 * fsv[1][1])
        right.append(fsv[1][2] / 3.4 + 1.5 * fsv[1][0] - 1.15 * fsv[1][1])
        right.append(fsv[1][2] / 3.4 - 1.5 * fsv[1][0] - 1.15 * fsv[1][1])
        right.append(fsv[1][2] / 3.4 - 1.5 * fsv[1][0] + 1.15 * fsv[1][1])

        for i in range(len(left)):
            left[i] = max(min(left[i], 25), 0)
            right[i] = max(min(right[i], 25), 0)
            newtonsLeft += left[i]
            newtonsRight += right[i]

        print('----------foot sensors----------')
        print('+ left ---- right +')
        print('+-------+ +-------+')
        print('|' + str(round(left[0], 1)) + '  ' + str(round(left[1], 1)) +
              '| |' + str(round(right[0], 1)) + '  ' + str(round(right[1], 1)) + '|  front')
        print('| ----- | | ----- |')
        print('|' + str(round(left[3], 1)) + '  ' + str(round(left[2], 1)) +
              '| |' + str(round(right[3], 1)) + '  ' + str(round(right[2], 1)) + '|  back')
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
        print('|' + str(ll) + '  ' + str(lr) + '| |' + str(rl) + '  ' + str(rr) + '|')
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
        # RGB leds
        for i in range(0, len(self.leds)):
            self.leds[i].set(rgb)
        # ears single-channel (blue)
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
        print('Keyboard works as before; additionally, control via WebSocket on ws://<host>:8765')
        print('JSON examples:')
        print('  {"type":"cmd","action":"forward"}')
        print('  {"type":"cmd","action":"led","rgb":"#00FF00"}')
        print('  {"type":"cmd","action":"hand","state":"open"}')
        print('  {"type":"get","sensor":"imu"}')

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
        for b in [self.lfootlbumper, self.lfootrbumper, self.rfootlbumper, self.rfootrbumper]:
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

        # shoulders (for demo motions)
        self.RShoulderPitch = self.getDevice("RShoulderPitch")
        self.LShoulderPitch = self.getDevice("LShoulderPitch")

        # keyboard
        self.keyboard = self.getKeyboard(); self.keyboard.enable(10 * self.timeStep)

    def __init__(self):
        Robot.__init__(self)
        self.currentlyPlaying = False
        self.findAndEnableDevices()
        self.loadMotionFiles()
        self.printHelp()
        # NEW: start websocket server (non-blocking)
        self._start_ws_server()

    # --------- APPLY COMMANDS FROM QUEUE (NEW) ----------
    def _apply_ws_command(self, cmd):
        """
        cmd is a dict already parsed from JSON.
        Supported:
          {"type":"cmd","action":"forward|backward|left|right|turn_left|turn_right"}
          {"type":"cmd","action":"led","rgb":"#RRGGBB"}
          {"type":"cmd","action":"hand","state":"open|close"}
          {"type":"get","sensor":"imu|gps|us|fsr"}
        """
        if not isinstance(cmd, dict):  # ignore
            return

        tp = cmd.get("type")
        if tp == "cmd":
            action = cmd.get("action", "")
            if action == "forward":
                self.startMotion(self.forwards)
            elif action == "backward":
                self.startMotion(self.backwards)
            elif action == "left":
                self.startMotion(self.sideStepLeft)
            elif action == "right":
                self.startMotion(self.sideStepRight)
            elif action == "turn_left":
                self.startMotion(self.turnLeft60)
            elif action == "turn_right":
                self.startMotion(self.turnRight60)
            elif action == "led":
                rgb = cmd.get("rgb", "#000000")
                if isinstance(rgb, str) and HEX_RGB.match(rgb):
                    rgb = rgb.lstrip("#")
                    val = int(rgb, 16)
                    self.setAllLedsColor(val)
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

    def run(self):
        # wave until first interaction (keyboard or ws command)
        self.handWave.setLoop(True)
        self.handWave.play()
        self.currentlyPlaying = self.handWave

        # wait for first event (non-blocking)
        while self.step(self.timeStep) != -1:
            # break if key pressed
            if self.keyboard.getKey() > 0:
                break
            # or break if any ws command arrived
            try:
                cmd = self.command_queue.get_nowait()
                # put it back for the main loop to handle uniformly
                self.command_queue.put(cmd)
                break
            except queue.Empty:
                pass

        self.handWave.setLoop(False)

        # main loop
        while self.step(self.timeStep) != -1:
            # 1) handle WebSocket commands (if any)
            try:
                while True:
                    cmd = self.command_queue.get_nowait()
                    self._apply_ws_command(cmd)
            except queue.Empty:
                pass

            # 2) also allow original keyboard controls
            key = self.keyboard.getKey()
            if key == Keyboard.LEFT:
                self.startMotion(self.sideStepLeft)
            elif key == Keyboard.RIGHT:
                self.startMotion(self.sideStepRight)
            elif key == Keyboard.UP:
                self.startMotion(self.forwards)
            elif key == Keyboard.DOWN:
                self.startMotion(self.backwards)
            elif key == Keyboard.LEFT | Keyboard.SHIFT:
                self.startMotion(self.turnLeft60)
            elif key == Keyboard.RIGHT | Keyboard.SHIFT:
                self.startMotion(self.turnRight60)
            elif key == ord('A'):
                self.printAcceleration()
            elif key == ord('G'):
                self.printGyro()
            elif key == ord('S'):
                self.printGps()
            elif key == ord('I'):
                self.printInertialUnit()
            elif key == ord('F'):
                self.printFootSensors()
            elif key == ord('B'):
                self.printFootBumpers()
            elif key == ord('U'):
                self.printUltrasoundSensors()
            elif key == ord('T'):
                self.startMotion(self.taiChi)
            elif key == ord('W'):
                self.startMotion(self.wipeForhead)
            elif key == Keyboard.HOME:
                self.printCameraImage(self.cameraTop)
            elif key == Keyboard.END:
                self.printCameraImage(self.cameraBottom)
            elif key == Keyboard.PAGEUP:
                self.setHandsAngle(0.96)
            elif key == Keyboard.PAGEDOWN:
                self.setHandsAngle(0.0)
            elif key == ord('7'):
                self.setAllLedsColor(0xff0000)
            elif key == ord('8'):
                self.setAllLedsColor(0x00ff00)
            elif key == ord('9'):
                self.setAllLedsColor(0x0000ff)
            elif key == ord('0'):
                self.setAllLedsColor(0x000000)
            elif key == ord('H'):
                self.printHelp()

            # 3) periodically push telemetry (optional)
            #    اینجا هر حلقه مقداری تله‌متری می‌فرستیم (می‌تونی با شمارنده کم‌تکرارترش کنی)
            self.telemetry_queue.put(self._ws_pack_telemetry())


# create the Robot instance and run main loop
robot = Nao()
robot.run()
