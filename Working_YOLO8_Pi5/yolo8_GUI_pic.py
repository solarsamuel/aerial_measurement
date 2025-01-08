import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import tkinter as tk
from tkinter import Label
from tkinter import filedialog
from PIL import Image, ImageTk
import math
from datetime import datetime

# Initialize the camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 1280)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load YOLOv8 model
model = YOLO("yolov8x_ncnn_model")

# Function to calculate changes in x, y, and diagonal pixels
def calculate_box_differences(box):
    x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates (x_min, y_min, x_max, y_max)
    dx = abs(x2 - x1)  # Change in x (width of the bounding box)
    dy = abs(y2 - y1)  # Change in y (height of the bounding box)
    diagonal = math.sqrt(dx**2 + dy**2)  # Diagonal length
    return dx, dy, diagonal

# Function to capture and process an image
def take_picture():
    global annotated_frame  # Declare as global to access in save function
    # Capture a frame from the camera
    frame = picam2.capture_array()
    
    # Run YOLO model on the captured frame
    results = model(frame)
    
    # Extract detected object class names and bounding box differences
    detected_objects = []
    box_differences = []
    if results[0].boxes:
        for box in results[0].boxes[:7]:  # Limit to the first 7 detected objects
            class_id = int(box.cls)  # Class ID for the detected object
            detected_objects.append(model.names[class_id])  # Map class ID to class name
            
            # Calculate changes in x, y, and diagonal pixels
            dx, dy, diagonal = calculate_box_differences(box)
            box_differences.append((dx, dy, diagonal))
    
    # Calculate average diagonal d for up to 7 cars
    if box_differences:
        avg_diagonal = sum([diagonal for _, _, diagonal in box_differences]) / len(box_differences)
    else:
        avg_diagonal = 0
    
    # Scale the average diagonal to 3 inches (assuming known scaling factor)
    inches = 3  # Target in inches
    pixels_per_inch = avg_diagonal / inches 
    
    # Update the label with detected objects and box differences
    if detected_objects:
        object_text = f"Objects detected: {', '.join(detected_objects)}"
        differences_text = "\n".join(
            [f"{obj}: Δx={dx:.1f}, Δy={dy:.1f}, Δd={diagonal:.1f}" 
             for obj, (dx, dy, diagonal) in zip(detected_objects, box_differences)]
        )
        avg_diagonal_text = f"Avg Δd : {avg_diagonal} pixels"
        pixels_per_inch_text = f"Pixels per inch: {pixels_per_inch:.1f}"
        detected_label.config(text=f"{object_text}\n{differences_text}\n{avg_diagonal_text}\n{pixels_per_inch_text}")
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

# Function to save the image with current datetime as filename
def save_image():
    if annotated_frame is not None:
        # Get current datetime for the filename
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{current_time}.png"
        
        # Save the annotated frame
        cv2.imwrite(filename, annotated_frame)
        print(f"Image saved as {filename}")

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

# Create and place the "Save" button
save_button = tk.Button(button_frame, text="Save Image", command=save_image)
save_button.pack(side=tk.LEFT, expand=True, padx=10)

# Create and place the "Quit" button
quit_button = tk.Button(button_frame, text="Quit", command=quit_app)
quit_button.pack(side=tk.RIGHT, expand=True, padx=10)

# Create and place an image label below the buttons
image_label = Label(root)
image_label.pack()

# Initialize global variable for annotated frame
annotated_frame = None

# Run the Tkinter event loop
root.mainloop()
