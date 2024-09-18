import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import pygame  # For playing background music and sound effects

# Initialize pygame mixer once
pygame.mixer.init()

# Load and play background music
pygame.mixer.music.load("resources/flat.mp3")  # Replace with your background music file path
pygame.mixer.music.play(-1)  # Loop the music indefinitely

# Load sound effects as separate Sound objects
win_sound_player1 = pygame.mixer.Sound("resources/player1.wav")
win_sound_player2 = pygame.mixer.Sound("resources/player2.wav")

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

# Import images
imgBackground = cv2.imread("resources/background1.png")
imgGameOver = cv2.imread("resources/game.jpg")
imgBall = cv2.imread("resources/ball.png", cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread("resources/bat2.png", cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread("resources/bat1.png", cv2.IMREAD_UNCHANGED)

# Ensure images have 4 channels (RGBA)
def ensure_alpha_channel(img):
    if img.shape[2] == 3:  # If image has 3 channels (RGB), add alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img

imgBall = ensure_alpha_channel(imgBall)
imgBat1 = ensure_alpha_channel(imgBat1)
imgBat2 = ensure_alpha_channel(imgBat2)

# Hand detector
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

# Variables
ballPos = [150, 180]
speedX = 30
speedY = 30
gameOver = False
winner = None
sound_played = False  # Flag to track if sound has been played
score = [0, 0]  # Score for Player 1 and Player 2

while True:
    success, img = cap.read()
    if not success:
        break  # Exit the loop if there's an issue with the camera
    img = cv2.flip(img, 1)
    imgRaw = img.copy()

    # Resize background to match the webcam feed size
    imgBackground_resized = cv2.resize(imgBackground, (img.shape[1], img.shape[0]))

    # Find hands
    hands, img = detector.findHands(img, draw=True, flipType=False)

    # Overlaying the background image with some transparency
    img = cv2.addWeighted(img, 0.2, imgBackground_resized, 0.8, 0)

    # Check for hands
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = imgBat1.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 415)

            if hand['type'] == 'Left':
                img = cvzone.overlayPNG(img, imgBat1, (56, y1))
                if 56 < ballPos[0] < 56 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] += 40
                    score[0] += 1  # Update score for Player 1

            if hand['type'] == 'Right':
                img = cvzone.overlayPNG(img, imgBat2, (1095, y1))
                if 1095 - 60 < ballPos[0] < 1095 and y1 - 30 < ballPos[1] < y1 + h1 + 30:
                     speedX = -speedX
                     ballPos[0] -= 40
                     score[1] += 1  # Update score for Player 2

    # Game over condition
    if ballPos[0] < 40 and not sound_played:
        gameOver = True
        winner = "Player 2"
        pygame.mixer.music.stop()  # Stop background music
        win_sound_player2.play()  # Play Player 2 win sound
        sound_played = True

    elif ballPos[0] > 1100 and not sound_played:
        gameOver = True
        winner = "Player 1"
        pygame.mixer.music.stop()  # Stop background music
        win_sound_player1.play()  # Play Player 1 win sound
        sound_played = True

    if gameOver:
        img = imgGameOver.copy()  # Use a copy to prevent modifying the original image
        cv2.putText(img, f"{winner} Wins!", (60, 40), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(img, f"Score: {score[0]} - {score[1]}", (65, 100), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 0), 2)
        cv2.putText(img, "Press 'r' to Restart", (90, 400), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)


        cv2.imshow("image", img)
        key = cv2.waitKey(1)
        if key == ord('r'):
            # Reset the game
            ballPos = [150, 180]
            speedX = 30  # Reset to original speed or desired value
            speedY = 30
            gameOver = False
            winner = None
            sound_played = False  # Reset sound played flag
            score = [0, 0]  # Reset scores

            # Restart background music
            pygame.mixer.music.load("resources/flat.mp3")  # Replace with your background music file path
            pygame.mixer.music.play(-1)  # Loop the music indefinitely
        continue  # Skip the rest of the loop until game is reset

    else:
        # Move the ball
        if ballPos[1] >= 550 or ballPos[1] <= 80:
            speedY = -speedY
        ballPos[0] += speedX
        ballPos[1] += speedY

    # Draw the ball
    img = cvzone.overlayPNG(img, imgBall, ballPos)

    # Display scores during the game
    cv2.putText(img, str(score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
    cv2.putText(img, str(score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)

    # Ensure the region to display raw camera feed exists
    if img.shape[0] >= 700 and img.shape[1] >= 233:
        img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break  # Exit the loop if 'q' is pressed

# Quit pygame mixer and release resources when done
pygame.mixer.music.stop()
pygame.quit()
cap.release()
cv2.destroyAllWindows()
