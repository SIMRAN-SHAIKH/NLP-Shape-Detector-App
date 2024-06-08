import streamlit as st
import cv2
import numpy as np
import mahotas
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import imutils

# Define function to perform shape detection
def describe_shapes(image):
    shapeFeatures = []
  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (13, 13), 0)
    thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)[1]
  
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.erode(thresh, None, iterations=2)
  
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
  
    for c in cnts:
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
  
        (x, y, w, h) = cv2.boundingRect(c)
        roi = mask[y:y + h, x:x + w]
  
        features = mahotas.features.zernike_moments(roi, cv2.minEnclosingCircle(c)[1], degree=8)
        shapeFeatures.append(features)
  
    return (cnts, shapeFeatures)

def main():
    st.title("Shape Detection")

    # Add file uploader for reference image
    ref_image = st.file_uploader("Upload Reference Image (reference.png)", type=["png", "jpg"])

    # Add file uploader for shapes image
    shapes_image = st.file_uploader("Upload Shapes Image (shapes.png)", type=["png", "jpg"])

    if ref_image and shapes_image:
        ref_image = cv2.imdecode(np.fromstring(ref_image.read(), np.uint8), 1)
        shapes_image = cv2.imdecode(np.fromstring(shapes_image.read(), np.uint8), 1)

        if st.button("Detect"):
            (_, gameFeatures) = describe_shapes(ref_image)
            (cnts, shapeFeatures) = describe_shapes(shapes_image)
            
            # Compute Euclidean distances between features
            D = dist.cdist(gameFeatures, shapeFeatures)
            i = np.argmin(D)
             
            # Loop over contours in shapes image
            for (j, c) in enumerate(cnts):
                if i != j:
                    box = cv2.minAreaRect(c)
                    box = np.intp(cv2.boxPoints(box))
                    cv2.drawContours(shapes_image, [box], -1, (0, 0, 255), 2)
              
            # Draw bounding box around detected shape
            box = cv2.minAreaRect(cnts[i])
            box = np.intp(cv2.boxPoints(box))
            cv2.drawContours(shapes_image, [box], -1, (0, 255, 0), 2)
            (x, y, w, h) = cv2.boundingRect(cnts[i])
            cv2.putText(shapes_image, "FOUND!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

            # Display images using Streamlit
            st.image(ref_image, caption='Reference Image', use_column_width=True)
            st.image(shapes_image, caption='Shapes Image with Detected Shape', use_column_width=True)

if __name__ == "__main__":
    main()
