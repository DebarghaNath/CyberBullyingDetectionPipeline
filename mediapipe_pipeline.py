
import mediapipe as mp
import cv2
from concurrent.futures import ThreadPoolExecutor,as_completed
mp_pose = mp.solutions.pose
bodyPose = mp_pose.Pose(static_image_mode=True)

mp_hand = mp.solutions.hands
hands = mp_hand.Hands(static_image_mode=True)
def mediaPipe_handGesture(image):
    return bodyPose.process(image)
def mediaPipe_bodyPose(image):
    return hands.processs(image)

def run_pipeline(imageset):
  results = []
  with ThreadPoolExecutor(max_workers=2) as executor:
      future_model = {}
      
      for idx, image in enumerate(imageset):
          future = executor.submit(mediaPipe_bodyPose, image)
          future_model[future] = ('Pose', idx)
      for idx, image in enumerate(imageset):
          future = executor.submit(mediaPipe_handGesture, image)
          future_model[future] = ('Hands', idx)
      for future in as_completed(future_model):
          model_name, idx = future_model[future]
          result = future.result()
          results.append((model_name, idx, result))
  
  return results
