######### no loggin warning on av ############
import av.logging
restore_default_callback = lambda *args: args
av.logging.restore_default_callback = restore_default_callback
av.logging.set_level(av.logging.ERROR)
##############################################

# tools 
import logging
import ssl
import uuid
import argparse
import json
# webapp
import asyncio
from aiohttp import web
# aiortc
from av import VideoFrame
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
# slam module
import numpy as np
import orbslam2

logger = logging.getLogger('RTCSLAM')
pcs = set()
relay = MediaRelay()

# encode numpy.ndarray to json
class NumpyArrayEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return JSONEncoder.default(self, obj)

# webrtc frame track handler
class SlamTrack(MediaStreamTrack):
  kind = 'video'

  def __init__(self, track, session, channels):
    super().__init__()  # don't forget this!
    self.track = track
    self.session = session
    self.channels = channels
    self.blank = VideoFrame.from_ndarray(np.zeros((480, 640, 3), np.uint8), format='rgb24')

  async def recv(self):
    frame = await self.track.recv()
    img = frame.to_ndarray(format='bgr24')
    new_frame = self.blank

    self.session.add_track(img)

    # get map frame
    img = self.session.get_frame()
    if img.size > 0: new_frame = VideoFrame.from_ndarray(img, format='bgr24')
    # push position
    position = self.session.get_position()
    state = self.session.tracking_state()
    try:
      self.channels['position'].send(json.dumps(position, cls=NumpyArrayEncoder))
      self.channels['state'].send(json.dumps(state, cls=NumpyArrayEncoder))
    except Exception as e:
      logger.error(str(e))
    # push features
    # features = self.session.get_features()
    # self.features_channel.send(json.dumps(features, cls=NumpyArrayEncoder))

    # push frames
    new_frame.pts = frame.pts
    new_frame.time_base = frame.time_base
    return new_frame

# webrtc session handler
async def session_handler(request):
  params = await request.json()
  offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])

  pc = RTCPeerConnection() 
  pc_id = 'WebRTC-Session(%s)' % uuid.uuid4()
  pcs.add(pc)

  def log_info(msg, *args):
    logger.info(pc_id + ' ' + msg, *args)
  
  # create slam session
  session = orbslam2.Session('../vocabulary/voc.bin', 640, 480, True)
  session.enable_viewer(off_screen = True)
  if params['mode'] == 'slam_save': 
    session.save_map(True, 'statis/map/test')
  elif params['mode'] == 'tracking': 
    session.load_map(True, 'static/map/test.yaml')

  # create data channel to send position and state
  channels = {
    'position' : pc.createDataChannel('position'),
    'state' : pc.createDataChannel('state')
  }

  @pc.on('connectionstatechange')
  async def on_connectionstatechange():
    log_info('Connection state is %s', pc.connectionState)
    if pc.connectionState == 'failed':
      session.release()
      await pc.close()
      pcs.discard(pc)

  # data channel messages
  @pc.on('datachannel')
  def on_datachannel(channel):
      @channel.on('message')
      def on_message(message):
        if isinstance(message, str) and message == 'release':
          session.release()

  # track images
  @pc.on('track')
  def on_track(track):
    log_info('Track %s received', track.kind)

    if track.kind == 'video':
      pc.addTrack(SlamTrack(relay.subscribe(track), session, channels))

    @track.on('ended')
    async def on_ended():
      log_info('Track %s ended', track.kind)
      session.release()

  # handle offer
  await pc.setRemoteDescription(offer)

  # send answer
  answer = await pc.createAnswer()
  await pc.setLocalDescription(answer)
  
  return web.json_response({
    'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type
  })

# close all rtc connection when shut down
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
  app.router.add_post('/session', session_handler)
  
  web.run_app(
    app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
  )