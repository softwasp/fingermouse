import cv2
import mediapipe as mp
import pyautogui
import time
import asyncio

async def move_mouse(cx, cy, screen_width, screen_height, w, h):
    #await asyncio.sleep(0.4)
    pyautogui.moveTo((cx) * screen_width / (w), (cy) * screen_height / (h))

async def main():
    start_time = None
    click_threshold = 1.0
    screen_width, screen_height = pyautogui.size()
    pyautogui.FAILSAFE = False
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_tracking_confidence=0.5, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    print(screen_width, screen_height)
    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)
        result = mp_hands.process(img)

        if result.multi_hand_landmarks:
            for id, lm, in enumerate(result.multi_hand_landmarks[0].landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * w)
                
        if result.multi_hand_landmarks:
            await move_mouse(cx, cy, screen_width, screen_height, w, h)
            
            hand_landmarks = result.multi_hand_landmarks[0]
            index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
            
            if index_finger_tip.y < middle_finger_tip.y and \
            index_finger_tip.y < ring_finger_tip.y and \
            index_finger_tip.y < thumb_tip.y:
            #            index_finger_tip.y < pinky_tip.y and \
                if start_time is None:
                    start_time = time.time()
                else:
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= click_threshold:
                        pyautogui.click()
                        print("click")
                        start_time = None 
            else:
                start_time = None

            mp_draw.draw_landmarks(img, result.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
        #cv2.imshow("fingerminger", img)
        cv2.waitKey(1)

asyncio.run(main())
