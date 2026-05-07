def get_eye_landmarks(face_landmarks, eye_indices, frame_width, fram_height):
    """
    Extract pixel coordinates of eye landmarks.
    """
    points=[]

    for idx in eye_indices:
        x=int(face_landmarks.landmark[idx].x* frame_width)
        y=int(face_landmarks.landmark[idx].y* fram_height)
        points.append((x, y))

    return points
