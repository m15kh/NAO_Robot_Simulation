import asyncio
import websockets

async def client():
    uri = "url-server"
    async with websockets.connect(uri) as websocket:
        await websocket.send("salam az Turkey!")
        reply = await websocket.recv()
        print("Server replied:", reply)

asyncio.run(client())
