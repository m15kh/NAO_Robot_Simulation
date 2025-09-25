import cv2
import time
import os
from ultralytics import YOLO

# -------- Config --------
CAM_INDEX = 2          # change if you have multiple webcams
FRAME_W = 1280
FRAME_H = 720
CONF_THRESH = 0.5
SMOOTHING = 0.75       # 0..1, higher = smoother motion of the center point
NEAR_FRACTION = 0.45   # bbox height >= this * frame_h => considered "close"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# -------- Helpers --------
def pick_main_person(result):
    """Return largest 'person' bbox (x1,y1,x2,y2,conf) or None."""
    if not hasattr(result, "boxes") or result.boxes is None or len(result.boxes) == 0:
        return None
    best = None
    best_area = 0
    for b in result.boxes:
        cls = int(b.cls[0])
        if cls != 0:               # class 0 = person in COCO
            continue
        conf = float(b.conf[0])
        if conf < CONF_THRESH:
            continue
        x1, y1, x2, y2 = map(float, b.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2, conf)
    return best

def decide_action(cx, h, frame_w, frame_h):
    """Left third -> TURN_LEFT, Right third -> TURN_RIGHT, Middle -> FORWARD/FORWARD by distance."""#BUG
    left_bound = frame_w / 3
    right_bound = 2 * frame_w / 3
    near = h >= frame_h * NEAR_FRACTION
    if cx < left_bound:
        return "TURN_LEFT"
    elif cx > right_bound:
        return "TURN_RIGHT"
    return "FORWARD" if near else "FORWARD" #BUG

def draw_overlay(frame, bbox, ema_cx, action, conf, fps, near_fraction):
    """Draw L/C/R guides, bbox, center dot, text, and a proximity bar."""
    h, w = frame.shape[:2]

    # thirds guide
    cv2.line(frame, (w//3, 0), (w//3, h), (0, 255, 255), 1)
    cv2.line(frame, (2*w//3, 0), (2*w//3, h), (0, 255, 255), 1)

    # proximity line
    near_y = int(h * (1.0 - near_fraction))  # higher fraction -> line is higher
    cv2.line(frame, (0, near_y), (w, near_y), (255, 255, 0), 1)
    cv2.putText(frame, "Near threshold", (10, max(near_y-6, 15)), FONT, 0.5, (255,255,0), 1, cv2.LINE_AA)

    # bbox + center
    if bbox is not None:
        x1, y1, x2, y2, conf_val = bbox
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        cx = int(ema_cx)
        cv2.circle(frame, (cx, int(h*0.5)), 6, (255, 0, 0), -1)
        cv2.putText(frame, f"person {conf_val:.2f}", (p1[0], max(p1[1]-8, 15)),
                    FONT, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # proximity bar (left edge)
        bbox_h = int(y2 - y1)
        frac = max(0.0, min(1.0, bbox_h / float(h)))
        bar_h = int(frac * h)
        cv2.rectangle(frame, (5, h - bar_h), (15, h - 1), (0, 200, 0), -1)
        cv2.rectangle(frame, (5, 5), (15, h - 1), (200, 200, 200), 1)
        cv2.putText(frame, "Proximity", (20, 20), FONT, 0.5, (200,200,200), 1, cv2.LINE_AA)

    # action + fps
    cv2.putText(frame, f"Action: {action}", (10, h - 20), FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, 30), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

def check_webcam_devices():
    """Check for available webcam devices in the system."""
    try:
        import glob
        devices = glob.glob('/dev/video*')
        if devices:
            print(f"Available video devices: {', '.join(devices)}")
            return devices
        else:
            print("No video devices found in /dev/video*")
            return []
    except Exception as e:
        print(f"Error checking for video devices: {e}")
        return []
        
def select_webcam():
    """Allow user to select which webcam to use."""
    devices = check_webcam_devices()
    
    if not devices:
        print("No webcam devices found.")
        return None
        
    print("\nSelect webcam to use:")
    for i, device in enumerate(devices):
        print(f"{i}: {device}")
    
    try:
        selection = int(input("\nEnter webcam number: "))
        if 0 <= selection < len(devices):
            return devices[selection]
        else:
            print(f"Invalid selection. Must be between 0 and {len(devices)-1}")
            return None
    except ValueError:
        print("Please enter a valid number")
        return None

def main():
    # Allow user to select webcam device
    selected_device = select_webcam()
    if not selected_device:
        print("No webcam selected. Exiting.")
        return
    
    try:
        model = YOLO("yolov8n.pt")  # small & fast
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Make sure the model file exists and dependencies are installed correctly.")
        return
    
    # Open the selected webcam device
    cap = None
    
    # First try to open the selected device directly
    print(f"Trying to open selected camera: {selected_device}")
    cap = cv2.VideoCapture(selected_device)
    
    # If direct device failed, try alternative methods
    if not cap or not cap.isOpened():
        print("Could not open selected device directly, trying alternative methods...")
        device_paths = [f'/dev/video{i}' for i in range(4)]  # Try video0 through video3
        print("Trying to open camera using direct device paths...")
        for device in device_paths:
            if os.path.exists(device):
                print(f"Trying device: {device}")
                cap = cv2.VideoCapture(device)
                if cap.isOpened():
                    print(f"Successfully opened camera at {device}")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
                    break
                else:
                    print(f"Device {device} exists but couldn't be opened")
    
    # If direct device path didn't work, try indices
    if not cap or not cap.isOpened():
        available_indices = list(range(4))  # Try indices 0, 1, 2, 3
        
        for idx in available_indices:
            print(f"Trying to open camera at index {idx}...")
            cap = cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            
            if cap.isOpened():
                print(f"Successfully opened camera at index {idx}")
                # Verify we can read a frame
                ret, test_frame = cap.read()
                if ret:
                    print("Successfully read a test frame from the camera")
                    break
                else:
                    print("Camera opened but couldn't read frame, trying next...")
                    cap.release()
            else:
                print(f"Failed to open camera at index {idx}")
    
    if not cap or not cap.isOpened():
        print("\nERROR: Could not open any webcam. Please check your camera connection.")
        print("You can try the following:")
        print("1. Run 'python release_webcam.py' to kill processes using the camera")
        print("2. Make sure the camera is connected and recognized by the system")
        print("3. Check camera permissions: ls -la /dev/video*")
        print("4. Try unplugging and reconnecting the camera")
        return

    ema_cx = None
    last_print = ""
    last_print_t = 0.0
    print_cooldown = 0.3

    t_prev = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: Failed to read frame.")
                break

            h, w = frame.shape[:2]

            # detect
            results = model.predict(source=frame, conf=CONF_THRESH, verbose=False)
            bbox = pick_main_person(results[0]) if results else None

            if bbox is None:
                action = "NO_HUMAN"
                conf = None
                ema_cx = None
            else:
                x1, y1, x2, y2, conf = bbox
                cx = 0.5 * (x1 + x2)
                bbox_h = (y2 - y1)

                # smooth center x
                ema_cx = cx if ema_cx is None else (SMOOTHING * ema_cx + (1 - SMOOTHING) * cx)
                action = decide_action(ema_cx, bbox_h, w, h)

            # FPS
            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # draw overlay
            draw_overlay(frame, bbox, ema_cx if ema_cx is not None else w/2, action, conf, fps, NEAR_FRACTION)

            # print to terminal (throttled for duplicates)
            if action != last_print or (t_now - last_print_t) > print_cooldown:
                print(action)
                last_print = action
                last_print_t = t_now

            # show
            cv2.imshow("Human Tracker", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
