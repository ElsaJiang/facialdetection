import cv2
import numpy as np
import time

ref_img = cv2.imread("martian.png", 0)

orb = cv2.ORB_create(nfeatures=1500)
kp1, des1 = orb.detectAndCompute(ref_img, None)

cap = cv2.VideoCapture("martian.MOV")
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    kp2, des2 = orb.detectAndCompute(gray, None)

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

                frame = cv2.rectangle(frame, (x, y), (x + size, y + size), (0,255,0), 3)

                # Text
                if int(time.time() * 2) % 2 == 0:
                    cv2.rectangle(frame, (30, 20), (600, 90), (0,0,255), -1)
                    cv2.putText(frame, "WE ARE NOT ALONE",
                                (40, 70),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (255,255,255), 3)

    cv2.imshow("Martian Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
