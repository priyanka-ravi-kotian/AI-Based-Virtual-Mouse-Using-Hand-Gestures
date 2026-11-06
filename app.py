import cv2
import pyautogui
import time
import math

from mediapipe.python.solutions import hands
from mediapipe.python.solutions import drawing_utils


# ============================================================
# 1. SCREEN SETUP
# ============================================================

screen_width, screen_height = pyautogui.size()

pyautogui.FAILSAFE = False


# ============================================================
# 2. MEDIAPIPE HAND DETECTION
# ============================================================

mp_hands = hands
mp_draw = drawing_utils

hand_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ============================================================
# 3. WEBCAM SETUP
# ============================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("AI Virtual Mouse", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AI Virtual Mouse", 1280, 720)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


# ============================================================
# 4. CLICK VARIABLES
# ============================================================

last_click_time = 0
last_right_click_time = 0
last_double_click_time = 0

pinch_active = False


# ============================================================
# 5. DRAG VARIABLES
# ============================================================

dragging = False
drag_start_time = 0

# Prevent accidental drag release
release_frames = 0


# ============================================================
# 6. SCROLL VARIABLES
# ============================================================

scroll_active = False
scroll_center_y = None

# Current gesture
current_gesture = "NO HAND"


# ============================================================
# 7. CURSOR SMOOTHING
# ============================================================

smoothening = 10

prev_x = 0
prev_y = 0

curr_x = 0
curr_y = 0

# Cursor stability
dead_zone = 3

# FPS calculation
fps = 0
prev_frame_time = time.time()


# ============================================================
# 8. ACTIVE CONTROL AREA
# ============================================================

frame_margin = 100


# ============================================================
# 9. MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # Capture webcam frame
    # --------------------------------------------------------

    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read webcam frame.")
        break

    # Calculate FPS
    current_frame_time = time.time()

    time_difference = (
        current_frame_time - prev_frame_time
    )

    if time_difference > 0:

        fps = 1 / time_difference

    prev_frame_time = current_frame_time


    # --------------------------------------------------------
    # Flip webcam horizontally
    # --------------------------------------------------------

    frame = cv2.flip(frame, 1)


    # --------------------------------------------------------
    # Convert BGR to RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # --------------------------------------------------------
    # Detect hand
    # --------------------------------------------------------

    results = hand_detector.process(rgb)


    # --------------------------------------------------------
    # Get frame dimensions
    # --------------------------------------------------------

    h, w, _ = frame.shape


    # ========================================================
    # 10. DRAW CONTROL AREA
    # ========================================================

    cv2.rectangle(
        frame,
        (frame_margin, frame_margin),
        (w - frame_margin, h - frame_margin),
        (255, 0, 255),
        2
    )


    # ========================================================
    # 11. HAND DETECTED
    # ========================================================

    current_gesture = "NO HAND"

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:
            current_gesture = "MOVE"


            # ------------------------------------------------
            # Draw hand landmarks
            # ------------------------------------------------

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


            # =================================================
            # 12. GET LANDMARKS
            # =================================================

            # Index finger tip = 8
            index_tip = hand_landmarks.landmark[8]

            # Thumb tip = 4
            thumb_tip = hand_landmarks.landmark[4]

            # Middle finger tip = 12
            middle_tip = hand_landmarks.landmark[12]


            # =================================================
            # 13. CONVERT TO PIXEL COORDINATES
            # =================================================

            index_x = int(index_tip.x * w)
            index_y = int(index_tip.y * h)

            thumb_x = int(thumb_tip.x * w)
            thumb_y = int(thumb_tip.y * h)

            middle_x = int(middle_tip.x * w)
            middle_y = int(middle_tip.y * h)


            # =================================================
            # 14. FINGER EXTENSION DETECTION
            # =================================================

            index_extended = (
                hand_landmarks.landmark[8].y
                < hand_landmarks.landmark[6].y
            )

            middle_extended = (
                hand_landmarks.landmark[12].y
                < hand_landmarks.landmark[10].y
            )

            ring_extended = (
                hand_landmarks.landmark[16].y
                < hand_landmarks.landmark[14].y
            )

            pinky_extended = (
                hand_landmarks.landmark[20].y
                < hand_landmarks.landmark[18].y
            )


            # Four fingers raised
            four_fingers_up = (
                index_extended
                and middle_extended
                and ring_extended
                and pinky_extended
            )

            if four_fingers_up:
                cv2.putText(
                    frame,
                    "FOUR FINGERS DETECTED",
                    (20, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )


            # =================================================
            # 15. DRAW FINGERTIP MARKERS
            # =================================================

            # Index
            cv2.circle(
                frame,
                (index_x, index_y),
                10,
                (0, 255, 0),
                cv2.FILLED
            )

            # Thumb
            cv2.circle(
                frame,
                (thumb_x, thumb_y),
                8,
                (255, 0, 255),
                cv2.FILLED
            )

            # Middle
            cv2.circle(
                frame,
                (middle_x, middle_y),
                8,
                (0, 255, 255),
                cv2.FILLED
            )


            # =================================================
            # 16. CALCULATE DISTANCES
            # =================================================

            # Thumb + Index
            # LEFT CLICK / DRAG

            distance = math.hypot(
                index_x - thumb_x,
                index_y - thumb_y
            )


            # Index + Middle
            # RIGHT CLICK

            right_click_distance = math.hypot(
                middle_x - index_x,
                middle_y - index_y
            )


            # Thumb + Middle
            # DOUBLE CLICK

            double_click_distance = math.hypot(
                middle_x - thumb_x,
                middle_y - thumb_y
            )


            # =================================================
            # 17. DRAW GESTURE LINES
            # =================================================

            # Thumb → Index
            cv2.line(
                frame,
                (thumb_x, thumb_y),
                (index_x, index_y),
                (255, 0, 255),
                2
            )


            # Index → Middle
            cv2.line(
                frame,
                (index_x, index_y),
                (middle_x, middle_y),
                (0, 255, 255),
                2
            )


            # =================================================
            # 18. CURRENT TIME
            # =================================================

            current_time = time.time()


            # =================================================
            # 19. SCROLL MODE
            # Four fingers raised
            # =================================================

            if four_fingers_up:

                current_gesture = "SCROLL"

                # ---------------------------------------------
                # If dragging, safely release mouse
                # ---------------------------------------------

                if dragging:

                    pyautogui.mouseUp()

                    dragging = False
                    pinch_active = False
                    drag_start_time = 0
                    release_frames = 0


                # ---------------------------------------------
                # Scroll center
                # ---------------------------------------------

                if scroll_center_y is None:

                    scroll_center_y = h // 2


                # ---------------------------------------------
                # Calculate distance from center
                # ---------------------------------------------

                scroll_difference = (
                    scroll_center_y - middle_y
                )


                # ---------------------------------------------
                # Dead zone
                #
                # Small movements around center
                # don't cause scrolling
                # ---------------------------------------------

                dead_zone = 35


                # ---------------------------------------------
                # SCROLL UP
                #
                # Hand is above center
                # ---------------------------------------------

                if scroll_difference > dead_zone:

                    # Calculate speed
                    scroll_speed = int(
                        scroll_difference / 10
                    )

                    # Minimum speed
                    scroll_speed = max(
                        1,
                        scroll_speed
                    )

                    # Maximum speed
                    scroll_speed = min(
                        10,
                        scroll_speed
                    )

                    pyautogui.scroll(
                        scroll_speed
                    )

                    current_gesture = "SCROLL UP"


                    cv2.putText(
                        frame,
                        "SCROLL UP",
                        (20, 210),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )


                # ---------------------------------------------
                # SCROLL DOWN
                #
                # Hand is below center
                # ---------------------------------------------

                elif scroll_difference < -dead_zone:

                    # Calculate speed
                    scroll_speed = int(
                        abs(scroll_difference) / 10
                    )

                    # Minimum speed
                    scroll_speed = max(
                        1,
                        scroll_speed
                    )

                    # Maximum speed
                    scroll_speed = min(
                        10,
                        scroll_speed
                    )

                    pyautogui.scroll(
                        -scroll_speed
                    )

                    current_gesture = "SCROLL DOWN"


                    cv2.putText(
                        frame,
                        "SCROLL DOWN",
                        (20, 210),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )


                # ---------------------------------------------
                # CENTER ZONE
                # ---------------------------------------------

                else:

                    current_gesture = "SCROLL READY"

                    cv2.putText(
                        frame,
                        "SCROLL READY",
                        (20, 210),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 0),
                        2
                    )


                # ---------------------------------------------
                # Draw scroll center line
                # ---------------------------------------------

                cv2.line(
                    frame,
                    (0, scroll_center_y),
                    (w, scroll_center_y),
                    (255, 255, 0),
                    2
                )


                scroll_active = True

            # =================================================
            # 20. NORMAL GESTURES
            # =================================================

            else:

                scroll_center_y = None
                scroll_active = False


                # =============================================
                # DOUBLE CLICK
                #
                # Thumb + Middle close
                # Index far away
                # =============================================

                if (
                    double_click_distance < 25
                    and distance > 60
                    and right_click_distance > 40
                ):

                    if (
                        current_time
                        - last_double_click_time
                        > 1.0
                    ):

                        pyautogui.doubleClick()

                        current_gesture = "DOUBLE CLICK"

                        last_double_click_time = (
                            current_time
                        )


                        cv2.putText(
                            frame,
                            "DOUBLE CLICK",
                            (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 0),
                            2
                        )


                # =============================================
                # RIGHT CLICK
                #
                # Index + Middle close
                # Thumb far away
                # =============================================

                elif (
                    right_click_distance < 25
                    and distance > 60
                    and double_click_distance > 40
                ):

                    if (
                        current_time
                        - last_right_click_time
                        > 1.0
                    ):

                        pyautogui.rightClick()

                        current_gesture = "RIGHT CLICK"

                        last_right_click_time = (
                            current_time
                        )


                        cv2.putText(
                            frame,
                            "RIGHT CLICK",
                            (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2
                        )


                # =============================================
                # THUMB + INDEX PINCH
                #
                # Quick pinch = LEFT CLICK
                # Hold pinch = DRAG
                # =============================================

                elif (
                    distance < 30
                    and right_click_distance > 40
                    and double_click_distance > 40
                ):

                    # -----------------------------------------
                    # Start pinch
                    # -----------------------------------------

                    if not pinch_active:

                        pinch_active = True

                        drag_start_time = (
                            current_time
                        )


                    # -----------------------------------------
                    # Hold pinch = DRAG
                    # -----------------------------------------

                    elif (
                        current_time
                        - drag_start_time
                        > 0.5
                    ):

                        if not dragging:

                            pyautogui.mouseDown()

                            dragging = True

                        current_gesture = "DRAGGING"


                        cv2.putText(
                            frame,
                            "DRAGGING",
                            (20, 170),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 165, 255),
                            2
                        )


                # =============================================
                # PINCH NOT DETECTED
                # =============================================

                else:

                    # -----------------------------------------
                    # If dragging, wait for stable release
                    # -----------------------------------------

                    if dragging:

                        release_frames += 1


                        # Require 5 consecutive frames
                        # before releasing
                        if release_frames >= 5:

                            pyautogui.mouseUp()

                            dragging = False
                            pinch_active = False
                            drag_start_time = 0
                            release_frames = 0


                            cv2.putText(
                                frame,
                                "DROP",
                                (20, 170),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2
                            )


                        else:

                            cv2.putText(
                                frame,
                                "DRAGGING",
                                (20, 170),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 165, 255),
                                2
                            )


                    # -----------------------------------------
                    # QUICK PINCH = LEFT CLICK
                    # -----------------------------------------

                    elif pinch_active:

                        if (
                            current_time
                            - drag_start_time
                            <= 0.5
                        ):

                            if (
                                current_time
                                - last_click_time
                                > 0.5
                            ):

                                pyautogui.click()

                                current_gesture = "LEFT CLICK"

                                last_click_time = (
                                    current_time
                                )


                                cv2.putText(
                                    frame,
                                    "LEFT CLICK",
                                    (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 255, 0),
                                    2
                                )


                        pinch_active = False
                        drag_start_time = 0


            # =================================================
            # 21. CURSOR MOVEMENT
            # =================================================

            # Don't move cursor using the index finger
            # while scrolling.

            if not four_fingers_up:

                screen_x = (
                    (index_x - frame_margin)
                    * screen_width
                    / (w - 2 * frame_margin)
                )

                screen_y = (
                    (index_y - frame_margin)
                    * screen_height
                    / (h - 2 * frame_margin)
                )


                # ---------------------------------------------
                # Keep cursor inside screen
                # ---------------------------------------------

                screen_x = max(
                    0,
                    min(
                        screen_width - 1,
                        screen_x
                    )
                )

                screen_y = max(
                    0,
                    min(
                        screen_height - 1,
                        screen_y
                    )
                )


                # =================================================
                # CURSOR STABILIZATION
                # =================================================

                # Calculate movement from previous position
                move_x = screen_x - prev_x
                move_y = screen_y - prev_y


                # -------------------------------------------------
                # Ignore extremely small movements
                # -------------------------------------------------

                if (
                    abs(move_x) < dead_zone
                    and abs(move_y) < dead_zone
                ):

                    screen_x = prev_x
                    screen_y = prev_y


                # -------------------------------------------------
                # Smooth cursor movement
                # -------------------------------------------------

                curr_x = (
                    prev_x
                    + (screen_x - prev_x)
                    / smoothening
                )

                curr_y = (
                    prev_y
                    + (screen_y - prev_y)
                    / smoothening
                )


                # -------------------------------------------------
                # Move mouse
                # -------------------------------------------------

                pyautogui.moveTo(
                    int(curr_x),
                    int(curr_y)
                )


                # -------------------------------------------------
                # Save position
                # -------------------------------------------------

                prev_x = curr_x
                prev_y = curr_y

    # ========================================================
    # STATUS PANEL
    # ========================================================

    panel_x = 10
    panel_y = 10
    panel_width = 300
    panel_height = 125

    # Panel background
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (
            panel_x + panel_width,
            panel_y + panel_height
        ),
        (30, 30, 30),
        -1
    )

    # Panel border
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (
            panel_x + panel_width,
            panel_y + panel_height
        ),
        (255, 255, 255),
        2
    )


    # Title
    cv2.putText(
        frame,
        "AI VIRTUAL MOUSE",
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (25, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


    # Current gesture
    cv2.putText(
        frame,
        f"Gesture: {current_gesture}",
        (25, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )


    # Hand detection status
    if results.multi_hand_landmarks:

        hand_status = "HAND: DETECTED"

    else:

        hand_status = "HAND: NOT DETECTED"


    cv2.putText(
        frame,
        hand_status,
        (25, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # ========================================================
    # GESTURE INSTRUCTION PANEL
    # ========================================================

    instruction_x = w - 300
    instruction_y = 10
    instruction_width = 285
    instruction_height = 235


    # Panel background
    cv2.rectangle(
        frame,
        (instruction_x, instruction_y),
        (
            instruction_x + instruction_width,
            instruction_y + instruction_height
        ),
        (30, 30, 30),
        -1
    )


    # Panel border
    cv2.rectangle(
        frame,
        (instruction_x, instruction_y),
        (
            instruction_x + instruction_width,
            instruction_y + instruction_height
        ),
        (255, 255, 255),
        2
    )


    # ========================================================
    # PANEL TITLE
    # ========================================================

    cv2.putText(
        frame,
        "GESTURE CONTROLS",
        (instruction_x + 15, instruction_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # GESTURE LIST
    # ========================================================

    instructions = [
        "Index Finger     : Move",
        "Thumb + Index    : Left Click",
        "Index + Middle   : Right Click",
        "Thumb + Middle   : Double Click",
        "Hold Pinch       : Drag",
        "Open Palm        : Scroll",
        "ESC              : Exit"
    ]


    start_y = instruction_y + 60

    for i, instruction in enumerate(instructions):

        cv2.putText(
            frame,
            instruction,
            (
                instruction_x + 12,
                start_y + i * 25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1
        )


    # ========================================================
    # 22. DISPLAY PROJECT TITLE
    # ========================================================

    cv2.putText(
        frame,
        "AI Virtual Mouse",
        (20, h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ========================================================
    # 23. DISPLAY EXIT MESSAGE
    # ========================================================

    cv2.putText(
        frame,
        "Press ESC to exit",
        (20, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # 24. SHOW WEBCAM
    # ========================================================

    cv2.imshow(
        "AI Virtual Mouse",
        frame
    )


    # ========================================================
    # 25. EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == 27:
        break


# ============================================================
# 26. CLEANUP
# ============================================================

if dragging:
    pyautogui.mouseUp()

cap.release()

cv2.destroyAllWindows()

print("AI Virtual Mouse stopped.")