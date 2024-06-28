import cv2
import numpy as np

original_image = cv2.imread('./asset/NU.jpg', cv2.IMREAD_COLOR)  
segment_image = cv2.imread('./asset/segment_0.jpg', cv2.IMREAD_COLOR)  

original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
segment_gray = cv2.cvtColor(segment_image, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()

keypoints_original, descriptors_original = orb.detectAndCompute(original_gray, None)
keypoints_segment, descriptors_segment = orb.detectAndCompute(segment_gray, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors_original, descriptors_segment)
matches = sorted(matches, key=lambda x: x.distance)

matched_points_original = np.float32([keypoints_original[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
matched_points_segment = np.float32([keypoints_segment[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

original_rect = cv2.boundingRect(matched_points_original)
segment_rect = cv2.boundingRect(matched_points_segment)

cv2.rectangle(original_image, (original_rect[0], original_rect[1]), (original_rect[0] + original_rect[2], original_rect[1] + original_rect[3]), (0, 255, 0), 2)
cv2.rectangle(segment_image, (segment_rect[0], segment_rect[1]), (segment_rect[0] + segment_rect[2], segment_rect[1] + segment_rect[3]), (0, 255, 0), 2)

cv2.imshow('Original Image with Matched Area', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
