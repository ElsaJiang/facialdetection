import cv2
import numpy as np
import time

# Load reference image
ref_img = cv2.imread("martian.png", 0)

orb = cv2.ORB_create(nfeatures=1500)
kp1, des1 = orb.detectAndCompute(ref_img, None)

cap = cv2.VideoCapture("martian.MOV")
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for consistent layout
    frame = cv2.resize(frame, (640, 360))
    raw_frame = frame.copy()
    detect_frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    kp2, des2 = orb.detectAndCompute(gray, None)

    detected = False

    if des2 is not None:
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) > 20:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                h, w = ref_img.shape
                pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts, M)

                points = np.int32(dst).reshape(-1, 2)
                x, y, w_box, h_box = cv2.boundingRect(points)
                size = max(w_box, h_box)

                # Draw square box on detection frame
                cv2.rectangle(detect_frame, (x, y), (x + size, y + size), (0,255,0), 3)

                detected = True

    # ---- RIGHT PANEL (status display) ----
    right_panel = np.zeros((720, 640, 3), dtype=np.uint8)

    if detected:
        if int(time.time() * 2) % 2 == 0:
            cv2.putText(right_panel, "WE ARE NOT ALONE",
                        (50, 360),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0,0,255), 3)
    else:
        cv2.putText(right_panel, "No Detection",
                    (150, 360),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,255,255), 2)

    # ---- LEFT SIDE (stack detection + raw) ----
    left_top = detect_frame
    left_bottom = raw_frame

    left_panel = np.vstack((left_top, left_bottom))

    # ---- FINAL GUI (left + right) ----
    final_display = np.hstack((left_panel, right_panel))

    cv2.imshow("Martian Detection GUI", final_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
