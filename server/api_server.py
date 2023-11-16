import argparse
import asyncio
import json
import logging
import ssl
import uuid
import orbslam2
import numpy as np

import cv2
from aiohttp import web
from av import VideoFrame
import av.logging

# monkey patch av.logging.restore_default_callback 
restore_default_callback = lambda *args: args
av.logging.restore_default_callback = restore_default_callback
av.logging.set_level(av.logging.ERROR)

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay


logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, session):
        super().__init__()  # don't forget this!
        self.track = track
        self.session = session
        self.blank = VideoFrame.from_ndarray(np.zeros((480, 640, 3), np.uint8), format="rgb24")

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        new_frame = self.blank

        self.session.add_track(img)
        img = self.session.get_frame()
        if img.size > 0:
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")

        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection() 
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    setattr(pc, 'session', None)
    setattr(pc, 'mode', params["mode"])
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            if pc.session is not None:
                pc.session.stop()
                pc.session = None
            await pc.close()
            pcs.discard(pc)

    # data channel messages
    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str):
                if message == 'stop' and pc.session is not None:
                    pc.session.stop()
                    pc.session = None
                elif message == 'position' and pc.session is not None:
                    matrix = np.ndarray() if pc.session is None else pc.session.position()
                    channel.send(json.dumps({"matrix": matrix}))

    # track images
    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if pc.session is None:
            pc.session = orbslam2.Session('../vocabulary/voc.bin', 640, 480, True)
            # set slam mode
            if pc.mode == 'slam_save':  pc.session.save_map(True, 'maps/map_test')
            elif pc.mode == 'tracking': pc.session.load_map(True, 'maps/map_test.yaml')

        if track.kind == "video":
            pc.addTrack(
                VideoTransformTrack(relay.subscribe(track), pc.session)
            )

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            if pc.session is not None:
                pc.session.stop()
                pc.session = None

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return web.json_response({
        "sdp": pc.localDescription.sdp, "type": pc.localDescription.type
    })


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC-SLAM API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8081, help="Port for HTTP server (default: 8081)")
    parser.add_argument("--cert", type=str, help="Path to the certificate file (for HTTPS)")
    parser.add_argument("--key", type=str, help="Path to the key file (for HTTPS)")
    parser.add_argument("--verbose", "-v", action="count")

    args = parser.parse_args()

    logging.basicConfig(level = (logging.DEBUG if args.verbose else logging.INFO))

    if args.cert:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert, args.key)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)
    
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )