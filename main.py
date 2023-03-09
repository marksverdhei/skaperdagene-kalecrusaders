import time
import cv2
import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark
import math as m

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
    return image[:, :width]

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

def sendWarningBadPosture(x):
    print("TODO: warning with sound")

def is_bad_posture(neck_inclination, torso_inclination ) -> bool:
    return neck_inclination < 40 and torso_inclination < 10

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


def main(double_camera=False):
    # If you stay in bad posture for more than 30 seconds send an alert.
    shouldNotBeInBadPostureForSeconds = 30

    # Should change position every
    shouldChangePositionEvery = 30
    old_l_shldr_y = None
        
    # Initialize frame counters for standing/sitting
    frames_without_changing_position = 0

    # Meta.
    fps = 1
    
    # Initialize frame counters for posture
    good_frames = 0
    bad_frames  = 0


    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
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

        _, width, _ = image.shape
        double_camera = False
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

        # to print the 33 landmarks, removed for this demo
        # mp_drawing.draw_landmarks(
        #     image,
        #     results.pose_landmarks,
        #     mp_pose.POSE_CONNECTIONS,
        #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.
        # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break

        if results.pose_landmarks == None :
            continue


        h, w = image.shape[:2]

        # Use lm and lmPose as representative of the following methods.
        lm = results.pose_landmarks
        lmPose  = mp_pose.PoseLandmark
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

        # Calculate distance between left shoulder and right shoulder points.
        offset = find_distance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
        
        # Assist to align the camera to point at the side view of the person.
        # Offset threshold 30 is based on results obtained from analysis over 100 samples.
        if offset < 100:
            cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
        else:
            cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)
        

        # Calculate angles.
        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
        
        print("neck: " + str(neck_inclination))
        print("torso: " + str(torso_inclination))

        # Draw landmarks.
        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)
        
        # Let's take y - coordinate of P3 100px above x1,  for display elegance.
        # Although we are taking y = 0 while calculating angle between P1,P2,P3.
        cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
        cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
        
        # Similarly, here we are taking y - coordinate 100px above x1. Note that
        # you can take any value for y, not necessarily 100 or 200 pixels.
        cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)
        
        # Put text, Posture and angle inclination.
        # Text string for display.
        angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

                    
        # Determine whether good posture or bad posture.
        # The threshold angles have been set based on intuition.
        if is_bad_posture(neck_inclination, torso_inclination):
            bad_frames = 0
            good_frames += 1
            
            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
            cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
            cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)
        
            # Join landmarks.
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)
        
        else:
            good_frames = 0
            bad_frames += 1
        
            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
            cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
            cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)
        
            # Join landmarks.
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)
        
        # Calculate the time of remaining in a particular posture.
        good_time = (1 / fps) * good_frames
        bad_time =  (1 / fps) * bad_frames
        
        # Pose time.
        if good_time > 0:
            time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
            print(time_string_good)
            cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
        else:
            time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
            print(time_string_bad)
            cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)
        
        if bad_time > shouldNotBeInBadPostureForSeconds:
            sendWarningBadPosture()

        # setting initial values
        if old_l_shldr_y == None: 
            old_l_shldr_y = l_shldr_y
        
        # checking if the person has changed position between standing / sitting
        ## checking that I have all the values or this check makes no sense
        if should_check_changed_position( l_shldr_y,  old_l_shldr_y, h):
            if has_change_position( l_shldr_y,  old_l_shldr_y, h):
                frames_without_changing_position = 0
                print("changed position")
            else:
                frames_without_changing_position += 1
                print("not changed position in " + str(frames_without_changing_position))

            if frames_without_changing_position > shouldChangePositionEvery:
                print("TODO: some alarm here")

            
        else:
            print("skipping position change")
        
        # update values
        old_l_shldr_y = l_shldr_y
        
        cv2.imshow('MediaPipe Pose',image)

        time.sleep(1 / fps)

    cap.release()


if __name__ == "__main__":
    main()