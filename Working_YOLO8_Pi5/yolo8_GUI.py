import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# Initialize the camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 1280)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load YOLOv8 model
model = YOLO("yolov8x_ncnn_model")

# Function to capture and process an image
def take_picture():
    # Capture a frame from the camera
    frame = picam2.capture_array()
    
    # Run YOLO model on the captured frame
    results = model(frame)
    
    # Extract detected object class names
    detected_objects = []
    if results[0].boxes:
        for box in results[0].boxes:
            class_id = int(box.cls)  # Class ID for the detected object
            detected_objects.append(model.names[class_id])  # Map class ID to class name
    
    # Update the label with detected objects
    if detected_objects:
        detected_label.config(
            text=f"Objects detected: {', '.join(detected_objects)}"
        )
    else:
        detected_label.config(text="No objects detected.")
    
    # Annotate the frame with detection results
    annotated_frame = results[0].plot()
    
    # Resize the frame to half its original dimensions
    height, width, _ = annotated_frame.shape
    resized_frame = cv2.resize(annotated_frame, (width // 2, height // 2))
    
    # Convert the frame to a format suitable for Tkinter
    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(img)
    
    # Update the image label
    image_label.config(image=img_tk)
    image_label.image = img_tk

# Function to quit the application
def quit_app():
    picam2.stop()
    root.destroy()

# Create the Tkinter window
root = tk.Tk()
root.title("YOLOv8 Object Detection")

# Set default window size (half of the camera frame size)
default_width = 1280 // 2
default_height = 1280 // 2 + 150  # Add extra space for labels and buttons
root.geometry(f"{default_width}x{default_height}")

# Create a label to display detected objects at the top
detected_label = Label(root, text="No objects detected.", font=("Arial", 12), fg="blue")
detected_label.pack(pady=5)

# Create a frame for the buttons
button_frame = tk.Frame(root)
button_frame.pack(fill=tk.X, pady=5)

# Create and place the "Take Picture" button
take_picture_button = tk.Button(button_frame, text="Take Picture", command=take_picture)
take_picture_button.pack(side=tk.LEFT, expand=True, padx=10)

# Create and place the "Quit" button
quit_button = tk.Button(button_frame, text="Quit", command=quit_app)
quit_button.pack(side=tk.RIGHT, expand=True, padx=10)

# Create and place an image label below the buttons
image_label = Label(root)
image_label.pack()

# Run the Tkinter event loop
root.mainloop()
