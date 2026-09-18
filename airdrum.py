import cv2
import numpy as np
import pygame 

# 1. Initialize Pygame Mixer and load sounds
pygame.mixer.init()
snare_sound = pygame.mixer.Sound('snare.wav')
cymbal_sound = pygame.mixer.Sound('cymbal.wav')

cap = cv2.VideoCapture(0)

lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])

# 2. State Flags to prevent audio stuttering (Debouncing)
snare_has_played = False
cymbal_has_played = False

while True:
    ret, frame = cap.read()
    if not ret: break
        
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    
    snare_box = [50, 150, 200, 300]
    cymbal_box = [width - 200, 150, width - 50, 300]
    
    cv2.rectangle(frame, (snare_box[0], snare_box[1]), (snare_box[2], snare_box[3]), (255, 0, 0), 3)
    cv2.putText(frame, "SNARE", (snare_box[0] + 20, snare_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    cv2.rectangle(frame, (cymbal_box[0], cymbal_box[1]), (cymbal_box[2], cymbal_box[3]), (255, 0, 0), 3)
    cv2.putText(frame, "CYMBAL", (cymbal_box[0] + 15, cymbal_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Reset tracking flags for this frame
    is_in_snare = False
    is_in_cymbal = False
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            cv2.circle(frame, (cx, cy), 15, (0, 255, 0), -1)
            
            # Check Snare Hit
            if snare_box[0] < cx < snare_box[2] and snare_box[1] < cy < snare_box[3]:
                is_in_snare = True
                cv2.rectangle(frame, (snare_box[0], snare_box[1]), (snare_box[2], snare_box[3]), (0, 0, 255), -1)
                
                # Play sound ONLY if it hasn't already played for this strike
                if not snare_has_played:
                    snare_sound.play()
                    snare_has_played = True
                    
            # Check Cymbal Hit
            elif cymbal_box[0] < cx < cymbal_box[2] and cymbal_box[1] < cy < cymbal_box[3]:
                is_in_cymbal = True
                cv2.rectangle(frame, (cymbal_box[0], cymbal_box[1]), (cymbal_box[2], cymbal_box[3]), (0, 255, 255), -1)
                
                if not cymbal_has_played:
                    cymbal_sound.play()
                    cymbal_has_played = True

    # 3. Reset logic: If the stick leaves the box, reset the flag so it can be hit again
    if not is_in_snare:
        snare_has_played = False
    if not is_in_cymbal:
        cymbal_has_played = False

    cv2.imshow("Virtual Air Drums", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()