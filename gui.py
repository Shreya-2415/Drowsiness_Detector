import tkinter as tk
from tkinter import Button, Label
import numpy as np
import cv2

def detect_drowsiness():
    face_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    EMOTIONS_LIST = ["Open eye", "Closed eye: Drowsiness"]
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Unable to open video capture.")
        return

    # Start the window thread for Qt backend
    cv2.startWindowThread()

    # Create a window with Qt backend
    cv2.namedWindow('Drowsiness Detection', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow('Drowsiness Detection', 800, 600)

    gap_between_predictions = 30  # Adjust this value to control the gap between predictions

    while True:
        ret, frame = video_capture.read()
        
        if not ret:
            print("Error: Unable to read frame.")
            break
        
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        combined_pred = ""

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Process each eye individually
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            
            # Detect left eye
            left_eye_gray = roi_gray[:, :w//2]
            # left_pred = EMOTIONS_LIST[np.random.randint(0, 2)]  # Random prediction for demonstration
            left_pred = EMOTIONS_LIST[1] if is_eye_open(left_eye_gray) else EMOTIONS_LIST[0]
            
            # Detect right eye
            right_eye_gray = roi_gray[:, w//2:]
            # right_pred = EMOTIONS_LIST[np.random.randint(0, 2)]  # Random prediction for demonstration
            right_pred = EMOTIONS_LIST[1] if is_eye_open(right_eye_gray) else EMOTIONS_LIST[0]

            # Combine predictions
            combined_pred = f"{left_pred} - {right_pred}"

        # Display combined prediction
        cv2.putText(frame, combined_pred, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

        cv2.imshow('Drowsiness Detection', frame)

        # Add a delay to slow down the rate of prediction display
        if cv2.waitKey(100) & 0xFF == ord('q'):  # Adjust the delay time (in milliseconds) as needed
            break

    video_capture.release()
    cv2.destroyAllWindows()

# def is_eye_open(eye_image):
#     # Convert the input image to grayscale if it's not already in grayscale
#     if len(eye_image.shape) > 2:
#         gray_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
#     else:
#         gray_eye = eye_image
    
#     # Apply Gaussian blur to reduce noise
#     blurred_eye = cv2.GaussianBlur(gray_eye, (5, 5), 0)
    
#     # Apply adaptive thresholding to binarize the image
#     _, thresh_eye = cv2.threshold(blurred_eye, 50, 255, cv2.THRESH_BINARY_INV)
    
#     # Find contours in the thresholded image
#     contours, _ = cv2.findContours(thresh_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # If no contours are found, consider the eye closed
#     if len(contours) == 0:
#         return False
    
#     # Otherwise, consider the eye open
#     return True

def is_eye_open(eye_image):
    # Convert the input image to grayscale if it's not already in grayscale
    if len(eye_image.shape) > 2:
        gray_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_eye = eye_image
    
    # Apply Gaussian blur to reduce noise
    blurred_eye = cv2.GaussianBlur(gray_eye, (5, 5), 0)
    
    # Apply adaptive thresholding to binarize the image
    _, thresh_eye = cv2.threshold(blurred_eye, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contours are found
    if len(contours) == 0:
        return False
    
    # Check if the area of the largest contour is below a certain threshold
    max_contour_area = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_contour_area) < 100:  # Adjust threshold as needed
        return False
    
    # Otherwise, consider the eye open
    return True



# GUI setup
top = tk.Tk()
top.geometry('800x600')
top.title('Drowsiness Detection')
top.configure(background='#CDCDCD')

heading = Label(top, text='Drowsiness Detection', pady=20, font=('arial', 25, 'bold'))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

detect_button = Button(top, text="Start Detection", command=detect_drowsiness, padx=10, pady=5)
detect_button.configure(background="#364156", foreground="white", font=('arial', 10, 'bold'))
detect_button.place(relx=0.4, rely=0.5)

top.mainloop()






