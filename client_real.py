# client_stream_pull.py
import asyncio, json, websockets, base64, time
import os
OUT_DIR = "frames"
OUTPUT_PATH = "output"
SAVE_DIR = os.path.join(OUTPUT_PATH, OUT_DIR)
FPS = 30

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
async def main():
    async with websockets.connect("ws://localhost:8765") as ws:
        hello = await ws.recv()
        print("HELLO:", hello)

        while True:
            await ws.send(json.dumps({"type":"get","sensor":"frame","camera":"top"}))

            data = json.loads(await ws.recv())
            if data.get("type") != "frame":
                continue

            fmt = data.get("format")
            b64 = data.get("data","")
            raw = base64.b64decode(b64)

            ts = int(time.time() * 1000)
            
            if fmt == "JPEG":
                path = f"output/{OUT_DIR}/frame_{ts}.jpg"
            else:
                path = f"output/{OUT_DIR}/frame_{ts}.{fmt.lower()}"

            with open(path, "wb") as f:
                f.write(raw)
            print("saved:", path)

            await asyncio.sleep(1.0 / FPS)

asyncio.run(main())
