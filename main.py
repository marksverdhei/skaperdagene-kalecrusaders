from collections import namedtuple
from enum import Enum
import time
import cv2
import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark
import math as m
import json
import pprint
import requests
from datetime import datetime

try:
    from desk_controller import DeskController
except ImportError:
    from mock_desk import DeskController

class DeskState(Enum):
    TOP = 1
    MIDDLE = 2
    BOTTOM = 3

USE_CAMERA = "right"

Point = namedtuple("Point", ["x", "y"])

# costants
# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# for the double camera
def split_image(image):
    "Divides the image in two, if the camera is double"
    _, width, _ = image.shape
    width = width // 2
    if USE_CAMERA == "left":
        return image[:, :width]
    else:
        return image[:, width:]

# helper functions for is_bad_posture

## find distance between two coordinates
def find_distance(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist
    
## Calculate angle between two coordinates
def findAngle(x1, y1, x2, y2):
    theta = m.acos( (y2 -y1)*(-y1) / (m.sqrt(
        (x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
    degree = int(180/m.pi)*theta
    return degree

def sendWarningBadPosture(desk):
    # shake_desk(desk, current_desk_position)
    desk.ascend_to_top()
    url = "https://api.particle.io/v1/events/Button?access_token=2213b7d75a8756282af2a2a25bb8b3e8856a7f2d"    
    # headers = {'Accept': 'text/event-stream'}
    s = requests.Session()
    with s.get(url, headers=None, stream=True, verify=False) as resp:
        for event in resp.iter_lines():
            if event:
                message = event.decode('utf8')
                if message == "event: Button":
                    desk.descend_to_half()
                    print("Button pressed: reset state")
                    break
                    

def is_bad_posture(neck_inclination, torso_inclination ) -> bool:
    return not(neck_inclination < 40 and torso_inclination < 10)

def should_check_changed_position(newShoulderY, oldShoulderY, height ) -> bool:
    if (newShoulderY != None and oldShoulderY == None) or (newShoulderY == None and oldShoulderY != None):
        # if the shoulder disappear/appears from screen then there was movement
        return True

    if (newShoulderY == None and oldShoulderY == None): 
        return True

    return height!= None and newShoulderY <= height and oldShoulderY <= height 

    
def has_change_position( newShoulderY, oldShoulderY, height ) -> bool:
    print(newShoulderY)    
    print(oldShoulderY)
    print(height)
    return abs(newShoulderY-oldShoulderY) > (height / 20)


def get_body_cords(results, h, w):
    lm = results.pose_landmarks
    lmPose  = mp.solutions.pose.PoseLandmark
    # Left shoulder.
    l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
    l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
    
    # Right shoulder.
    r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
    r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
    
    # Left ear.
    l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
    l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
    
    # Left hip.
    l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
    l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

    return [
        Point(
            l_shldr_x,
            l_shldr_y
        ),
        Point(
            r_shldr_x,
            r_shldr_y
        ),
        Point(
            l_ear_x,
            l_ear_y
        ),
        Point(
            l_hip_x,
            l_hip_y
        ),
    ]


def toggle_desk(desk, current_desk_position):
    if current_desk_position == DeskState.TOP:
        desk.descend_to_bottom()
        return DeskState.BOTTOM
    # elif current_desk_position == DeskState.BOTTOM :
    else:
        desk.ascend_to_top()
        return DeskState.TOP


def shake_desk(desk, current_desk_position) :
    if current_desk_position == DeskState.TOP:
        desk.descend_to_half()
        desk.ascend_to_top()
    
    if current_desk_position == DeskState.BOTTOM:
        desk.ascend_to_half()
        desk.descend_to_bottom()

    if current_desk_position == DeskState.MIDDLE:
        desk.ascend_to_top()
        desk.descend_to_half()
        
def main(double_camera=False):
    desk = DeskController()
    # desk.ascend_to_top()
    bad_time_start = datetime.now()
    good_time_start = datetime.now()

    current_desk_position = DeskState.TOP

    
    # If you stay in bad posture for more than 10 seconds send an alert.
    bad_posture_alert_threshold_seconds = 10

    # Should change position every
    # shouldChangePositionEvery = 30
    shouldChangePositionEvery = float("inf")

    old_l_shldr_y = None

    # Should leave desk every 
    # shouldChangeLeaveDeskEvery = 180
    shouldChangeLeaveDeskEvery = float("inf")

        
    # Initialize frame counters for standing/sitting
    last_time_changed_position = datetime.now()

    # Initialize frame counters for posture
    mp_pose = mp.solutions.pose

    # For webcam input:
    cap = cv2.VideoCapture(0)
    model = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    while cap.isOpened():
        success, image = cap.read()
        
        if not success:
            print("Ignoring empty camera frame.")
            continue

        if double_camera:
            image = split_image(image)


        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model.process(image)


        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        if cv2.waitKey(5) & 0xFF == 27:
            break

        if results.pose_landmarks == None :
            continue

        h, w = image.shape[:2]
        (
            left_shoulder_cords,
            right_shoulder_cords,
            ear_cords,
            hip_cords
        ) = get_body_cords(results, h, w)

        # Calculate distance between left shoulder and right shoulder points.
        offset = find_distance(
            left_shoulder_cords.x,
            left_shoulder_cords.y,
            right_shoulder_cords.x,
            left_shoulder_cords.y
        )
        
        # Assist to align the camera to point at the side view of the person.
        # Offset threshold 30 is based on results obtained from analysis over 100 samples.
        if offset < 100:
            cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
        else:
            cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)
        

        # Calculate angles.
        neck_inclination = findAngle(*left_shoulder_cords, *ear_cords)
        torso_inclination = findAngle(*hip_cords, *left_shoulder_cords)
        
        print("neck: " + str(neck_inclination))
        print("torso: " + str(torso_inclination))

        # Draw landmarks.h
        cv2.circle(image, left_shoulder_cords, 7, yellow, -1)
        cv2.circle(image, ear_cords, 7, yellow, -1)
        
        # Let's take y - coordinate of P3 100px above x1,  for display elegance.
        # Although we are taking y = 0 while calculating angle between P1,P2,P3.
        cv2.circle(image, (left_shoulder_cords.x, left_shoulder_cords.y - 100), 7, yellow, -1)
        cv2.circle(image, right_shoulder_cords, 7, pink, -1)
        cv2.circle(image, hip_cords, 7, yellow, -1)
        
        # Similarly, here we are taking y - coordinate 100px above x1. Note that
        # you can take any value for y, not necessarily 100 or 200 pixels.
        cv2.circle(image, (hip_cords.x, hip_cords.y - 100), 7, yellow, -1)
        
        # Put text, Posture and angle inclination.
        # Text string for display.
        angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

                    
        # Determine whether good posture or bad posture.
        # The threshold angles have been set based on intuition.
        now = datetime.now()
        if is_bad_posture(neck_inclination, torso_inclination):
            good_time_start = now # no bad time
            label_color = red
        else:
            bad_time_start = now # no good time
            label_color = green
        
        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, label_color, 2)
        cv2.putText(image, str(int(neck_inclination)), (left_shoulder_cords.x + 10, left_shoulder_cords.y), font, 0.9, label_color, 2)
        cv2.putText(image, str(int(torso_inclination)), (hip_cords.x + 10, hip_cords.y), font, 0.9, label_color, 2)
    
        # Join landmarks.
        cv2.line(image, left_shoulder_cords, ear_cords, label_color, 4)
        cv2.line(image, left_shoulder_cords, (left_shoulder_cords.x, left_shoulder_cords.y - 100), label_color, 4)
        cv2.line(image, hip_cords, left_shoulder_cords, label_color, 4)
        cv2.line(image, hip_cords, (hip_cords.x, hip_cords.y - 100), label_color, 4)
        
        # Calculate the time of remaining in a particular posture.
        good_time = now - good_time_start
        bad_time = now - bad_time_start

        # Pose time.
        if good_time.total_seconds() > 0:
            time_string_good = 'Good Posture time : ' + str(round(good_time.total_seconds(), 1)) + ''
            print(time_string_good)
            cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
        else:
            time_string_bad = 'Bad Posture time : ' + str(round(bad_time.total_seconds(), 1)) + ''
            print(time_string_bad)
            cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)
        
        if bad_time.total_seconds() > bad_posture_alert_threshold_seconds:
            sendWarningBadPosture(desk, current_desk_position)
            current_desk_position = DeskState.MIDDLE
            bad_time_start = now
            good_time_start = now
            last_time_changed_position = now

        # setting initial values
        if old_l_shldr_y == None: 
            old_l_shldr_y = left_shoulder_cords.y
        
        # checking if the person has changed position between standing / sitting
        ## checking that I have all the values or this check makes no sense
        if should_check_changed_position(left_shoulder_cords.y,  old_l_shldr_y, h):
            if has_change_position(left_shoulder_cords.y,  old_l_shldr_y, h):
                last_time_changed_position = now
                print("changed position")
                cv2.putText(image, "changed position", (10, h - 40), font, 0.9, green, 2)
        
            else:
                print("not changed position in " + str(last_time_changed_position))
                cv2.putText(image, "not changed position in" + str(last_time_changed_position), (10, h - 40), font, 0.9, yellow, 2)
            if (now - last_time_changed_position).total_seconds() > shouldChangePositionEvery:
                print("change desk position")
                current_desk_position = toggle_desk(desk, current_desk_position)
                cv2.putText(image, "MUST CHANGE POSITION", (10, h - 40), font, 0.9, red, 2)
        else:
            print("skipping position change")
        
        # update values
        old_l_shldr_y = left_shoulder_cords.y
        cv2.imshow('MediaPipe Pose',image)

    cap.release()
    # desk.close()

if __name__ == "__main__":
    main(double_camera=True)