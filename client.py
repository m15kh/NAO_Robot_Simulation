# client_save_jpeg.py
import asyncio, json, websockets, base64

OUT = "frame.jpg"

async def main():
    async with websockets.connect("ws://localhost:8765") as ws:
        hello = await ws.recv()
        print("HELLO:", hello)

        # ask server for one frame (top camera)
        await ws.send(json.dumps({"type":"get","sensor":"frame","camera":"top"}))

        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if data.get("type") == "frame":
                fmt = data.get("format")
                b64 = data.get("data", "")
                raw = base64.b64decode(b64)

                if fmt == "JPEG":
                    with open(OUT, "wb") as f:
                        f.write(raw)
                    print(f"Saved JPEG -> {OUT}")
                else:
                    print(f"Got non-JPEG format = {fmt}. Use client B.")
                break
            


asyncio.run(main())
