import cv2
import sys
import orbslam2
import time


def load_image(img_folder, img_ts_path):
  imgs = []
  ts = []
  
  with open(img_ts_path, 'r') as f:
    ts = f.read().strip().split()
  
  for(_, t) in enumerate(ts):
    img_path = img_folder + '/' + t + '.png'
    imgs.append(img_path)
  
  return imgs, ts

# python mono_euroc.py ../vocabulary/voc.bin configs/EuRoc.yaml ~/dataset/mav0/cam0/data configs/EuRoc_ts/V101.txt

if __name__ == '__main__' :
  voc, conf, img_folder, img_ts = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
  
  imgs, ts = load_image(img_folder, img_ts)
  
  session = orbslam2.Session(voc, conf, True)
  session.enable_viewer(conf)
  
  input('press when pangolin window is active')
  
  for i in range(0, len(imgs)):
    img = cv2.imread(imgs[i])
    session.add_track(img, float(ts[i]))
    time.sleep(1/30)
    
  input('press to quit')
  session.stop()