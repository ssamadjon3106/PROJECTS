import pygame
pygame.mixer.init()
ALARM_SOUND=pygame.mixer.Sound('/Users/samadjon/Documents/projects/Real-Time Drowsiness Detection System/src/alarm.mp3')
def play_alarm():
    if not pygame.mixer.get_busy():
        ALARM_SOUND.play()
def stop_alarm():
    pygame.mixer.stop()