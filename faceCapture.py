import numpy as np
import matplotlib.pyplot as plt
import cv2


# final image size parameters
cheight = 100
cwidth = 100

# webcam capture parameters
width = 640
height = 480

filenamePrefix = "student"  # this is the prefix for the image filenames

print("Press 's' to save, 'q' to quit")

cap = cv2.VideoCapture(0)
cap.set(3, width)  # set the width
cap.set(4, height)  # set the height

# markers for positioning eyes
yeye = 200
xeye1 = 288
xeye2 = 365

# markers for crop
ycrop1 = 140
xcrop1 = 220
xcrop2 = 440
ycrop2 = 350

count = 0

while True:
    # Capture frame-by-frame
    (ret, frame) = cap.read()

    # convert to greyscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # determine actual dimensions
    (height, width) = grey.shape

    mgrey = np.copy(grey)

    # draw markers for positioning eyes
    cv2.circle(mgrey, (xeye1, yeye), 8, (255, 255, 0), 2)
    cv2.circle(mgrey, (xeye2, yeye), 8, (255, 255, 0), 2)

	# draw rectangular to indicate crop
    cv2.rectangle(mgrey, (xcrop1, ycrop1), (xcrop2, ycrop2), 255)

    # Display the resulting frame
    wtitle = "Capturing face for " + filenamePrefix
    cv2.imshow(wtitle, mgrey)

	# wait for key press
    key = cv2.waitKey(1)

	# save the image to file if user presses 's'
    if key == ord('s'):
        # crop to create square image
        cgrey = grey[ycrop1:ycrop2, xcrop1:xcrop2]

        resGrey = cv2.resize(cgrey, (cwidth, cheight))
        filename = "faces_client/" + filenamePrefix + str(count) + ".png"
        cv2.imwrite(filename, resGrey)

        print("Face captured as ", filename)
        count += 1

    # quit program if users presses 'q'
    if key == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
