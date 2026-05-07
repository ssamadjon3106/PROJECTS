import numpy as np

# MediaPipe eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def calculate_EAR(eye_points):
    A=np.linalg.norm(np.array(eye_points[1])-np.array(eye_points[5]))
    B=np.linalg.norm(np.array(eye_points[2])-np.array(eye_points[4]))
    C=np.linalg.norm(np.array(eye_points[0])-np.array(eye_points[3]))

    ear=(A+B)/(2.0*C)

    return ear