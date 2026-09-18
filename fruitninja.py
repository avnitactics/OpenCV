import cv2
import numpy as np
import random
import math

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define HSV range for tracking a BLUE object (the "sword")
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

# Game Variables
score = 0
fruits = [] # List to hold our falling fruits
spawn_rate = 30 # Spawn a new fruit every X frames
frame_count = 0

# Trail for the sword slash effect
sword_trail = []
max_trail_length = 10

def spawn_fruit(frame_width):
    """Creates a new fruit dictionary with random properties."""
    x = random.randint(50, frame_width - 50)
    # Give them random colors (BGR format) and slightly varying speeds
    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    speed = random.randint(5, 12)
    return {"x": x, "y": 0, "radius": 30, "color": color, "speed": speed, "active": True}

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Flip to act like a mirror
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    
    # 1. Masking & Tracking the Sword
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    sword_pos = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 500: # Only track if it's large enough
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                sword_pos = (cx, cy)
                
                # Add to trail and keep it short
                sword_trail.append(sword_pos)
                if len(sword_trail) > max_trail_length:
                    sword_trail.pop(0)

    # Draw the sword trail for a cool "slash" visual
    if len(sword_trail) > 1:
        for i in range(1, len(sword_trail)):
            thickness = int(np.sqrt(max_trail_length / float(max_trail_length - i + 1)) * 5)
            cv2.line(frame, sword_trail[i - 1], sword_trail[i], (255, 255, 255), thickness)

    # 2. Game Logic: Spawning Fruits
    frame_count += 1
    if frame_count % spawn_rate == 0:
        fruits.append(spawn_fruit(width))
        # Slowly increase difficulty by spawning faster
        if spawn_rate > 10: 
            spawn_rate -= 1 

    # 3. Game Logic: Updating & Slicing Fruits
    for fruit in fruits:
        if not fruit["active"]: continue
        
        # Move fruit down
        fruit["y"] += fruit["speed"]
        
        # Draw the fruit
        cv2.circle(frame, (fruit["x"], fruit["y"]), fruit["radius"], fruit["color"], -1)
        # Draw a little stem
        cv2.line(frame, (fruit["x"], fruit["y"] - fruit["radius"]), (fruit["x"], fruit["y"] - fruit["radius"] - 10), (0, 100, 0), 3)

        # Collision Detection: Euclidean Distance
        if sword_pos:
            dist = math.sqrt((sword_pos[0] - fruit["x"])**2 + (sword_pos[1] - fruit["y"])**2)
            
            # If the sword center is inside the fruit radius, it's a hit!
            if dist < fruit["radius"] + 15: # 15 is the estimated sword tip radius
                fruit["active"] = False
                score += 10
                # Visual feedback: flash a white hit marker
                cv2.circle(frame, (fruit["x"], fruit["y"]), fruit["radius"] + 20, (255, 255, 255), 5)
                
        # Deactivate fruits that fall off screen
        if fruit["y"] > height:
            fruit["active"] = False

    # Clean up inactive fruits to save memory
    fruits = [f for f in fruits if f["active"]]

    # 4. Display UI
    cv2.putText(frame, f"SCORE: {score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
    
    cv2.imshow("Fruit Ninja OpenCV", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()