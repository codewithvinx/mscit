import cv2
import glob
import numpy as np


size = (9, 6)

objp = np.zeros((size[0]*size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)


obj_points = []
img_points = []


images = glob.glob("cameraCalibrationImages/*")

print("Images found:", len(images))


for file in images:

    img = cv2.imread(file)

    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    found, corners = cv2.findChessboardCorners(
        gray,
        size
    )


    print(file, "Corners found:", found)


    if found:
        obj_points.append(objp)
        img_points.append(corners)



print("Valid images:", len(obj_points))


if len(obj_points) >= 3:


    ret, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        obj_points,
        img_points,
        gray.shape[::-1],
        None,
        None
    )


    print("\nCamera Matrix:")
    print(camera_matrix)

    print("\nDistortion:")
    print(distortion)



    # Use existing image as test image
    test = cv2.imread(
        "cameraCalibrationImages/OpenCV_Chessboard.png"
    )


    corrected = cv2.undistort(
        test,
        camera_matrix,
        distortion
    )


    cv2.imwrite(
        "corrected.jpg",
        corrected
    )


    cv2.imshow("Original", test)
    cv2.imshow("Corrected", corrected)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    print("Calibration completed")


else:

    print("Need at least 3 chessboard images")
