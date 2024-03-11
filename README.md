# SIFT Feature Detection and Homography to combine Images with different prespective into one single paranomic image.
The images used in the code are uploaded to the Image folder, Same are shown below.
![Image/PA120272.JPG](https://github.com/HKyatham/SIFT_Detection_and_Homography_for_stiching_Images/blob/main/Images/PA120272.JPG)

![Image/PA120273.JPG](https://github.com/HKyatham/SIFT_Detection_and_Homography_for_stiching_Images/blob/main/Images/PA120273.JPG)

![Image/PA120274.JPG](https://github.com/HKyatham/SIFT_Detection_and_Homography_for_stiching_Images/blob/main/Images/PA120274.JPG)

![Image/PA120275.JPG](https://github.com/HKyatham/SIFT_Detection_and_Homography_for_stiching_Images/blob/main/Images/PA120275.JPG)

SIFT Feature detection is performed on the images to detect various common features between 2 images.
```
  sift = cv2.SIFT_create()
  kp1, des1 = sift.detectAndCompute(frame1, None)
  kp2, des2 = sift.detectAndCompute(frame2, None)
```

Brute Force matcher is used to match features.
```
  bf = cv2.BFMatcher()
  matches = bf.knnMatch(des1, des2, k=2)
```

Features are highlighted and lines are drawn while showing features side by side.
```
  img_matches = cv2.drawMatches(frame1, kp1, frame2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
```

Feature Highlighted Image is shown below.
![Image/Feature.png](https://github.com/HKyatham/SIFT_Detection_and_Homography_for_stiching_Images/blob/main/Images/Feature.png)

Homography is calculated on various matched features in both images.
```
  src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
  dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
  H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```

Images are combined using cv2 command to wrap the prespective.
```
  result = cv2.warpPerspective(frame1, H, (frame1.shape[1] + frame2.shape[1], frame1.shape[0]))
  result[0:frame2.shape[0], 0:frame2.shape[1]] = frame2
```

Combined result is shown below.
![Image/Combined_2_Images.png](https://github.com/HKyatham/SIFT_Detection_and_Homography_for_stiching_Images/blob/main/Images/Combined_2_Images.png)
