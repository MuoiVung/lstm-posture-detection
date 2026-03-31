import urllib.request
import cv2
import numpy as np
from app.services.pose_service import get_pose_service
from app.services.feature_service import engineer_features, landmarks_to_features

# download a typical sitting pose image
url = "https://images.unsplash.com/photo-1498050108023-c5249f4df085?auto=format&fit=crop&q=80&w=600"
req = urllib.request.urlopen(url)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr, -1)

ps = get_pose_service()
_, bts = cv2.imencode('.jpg', img)
res = ps.extract_landmarks(bts.tobytes())

if res:
    lm = res["key_landmarks"]
    print("MediaPipe Raw extraction:")
    print("Nose Z:", lm["nose"][2])
    feats = engineer_features(lm)
    print("Engineered features:")
    for k, v in feats.items():
        if k in ['torso_inclination', 'left_shoulder_hip_knee_angle', 'shoulder_tilt_angle']:
            print(f"  {k}: {v:.2f}")

    print("Normalized nose Y:", (lm['nose'] - (lm['left_hip']+lm['right_hip'])/2)[1] / np.linalg.norm(lm['left_shoulder'] - lm['right_shoulder']))
else:
    print("No pose found!")
