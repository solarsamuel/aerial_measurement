import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import tkinter as tk
from tkinter import Label
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

# Create the Tkinter window
root = tk.Tk()
root.title("YOLOv8 Object Detection and Line Drawing")

# Set default window size (half of the camera frame size)
default_width = 1280 // 2
default_height = 1280 // 2 + 150  # Add extra space for labels and buttons
root.geometry(f"{default_width}x{default_height}")

# Create a label to display detected objects and distance at the top
detected_label = Label(root, text="No objects detected.", font=("Arial", 12), fg="blue")
detected_label.pack(pady=5)

# Create a frame for the buttons
button_frame = tk.Frame(root)
button_frame.pack(fill=tk.X, pady=5)

# Create and place the "Take Picture" button
take_picture_button = tk.Button(button_frame, text="Take Picture", command=lambda: take_picture())
take_picture_button.pack(side=tk.LEFT, expand=True, padx=10)

# Create and place the "Save" button
save_button = tk.Button(button_frame, text="Save Image", command=lambda: save_image())
save_button.pack(side=tk.LEFT, expand=True, padx=10)

# Create and place the "Quit" button
quit_button = tk.Button(button_frame, text="Quit", command=root.quit)
quit_button.pack(side=tk.RIGHT, expand=True, padx=10)

# Create and place an image label below the buttons
image_label = Label(root)
image_label.pack()

# Initialize global variable for annotated frame and points for line drawing
annotated_frame = None
click_points = []

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

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
    car_and_phone_differences = []
    if results[0].boxes:
        for box in results[0].boxes[:7]:  # Limit to the first 7 detected objects
            class_id = int(box.cls)  # Class ID for the detected object
            object_name = model.names[class_id]
            detected_objects.append(object_name)  # Map class ID to class name
            
            # Calculate changes in x, y, and diagonal pixels
            dx, dy, diagonal = calculate_box_differences(box)
            box_differences.append((dx, dy, diagonal))
            
            # Filter for car and cell phone objects
            if object_name.lower() in ["car", "cell phone"]:
                car_and_phone_differences.append(diagonal)
            
    # Calculate the average delta d for car and cell phone objects (up to 7)
    if car_and_phone_differences:
        avg_car_phone_diagonal = sum(car_and_phone_differences) / len(car_and_phone_differences)
    else:
        avg_car_phone_diagonal = 0
    
    # Update the label with detected objects and box differences
    if detected_objects:
        object_text = f"Objects detected: {', '.join(detected_objects)}"
        differences_text = "\n".join(
            [f"{obj}: Δx={dx:.1f}, Δy={dy:.1f}, Δd={diagonal:.1f}" 
             for obj, (dx, dy, diagonal) in zip(detected_objects, box_differences)]
        )
        avg_car_phone_text = f"Avg Δd (Car/Cell phone): {avg_car_phone_diagonal:.1f} pixels"
        pixels_per_inch_text = f"Pixels per inch: {avg_car_phone_diagonal / 3:.1f}"  # Assuming 3 inches is the reference
        detected_label.config(text=f"{object_text}\n{differences_text}\n{avg_car_phone_text}\n{pixels_per_inch_text}")
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

# Function to handle mouse click events and draw lines
def handle_click(event):
    global click_points
    
    # Store the clicked point
    click_points.append((event.x, event.y))
    
    # If two points are clicked, draw a line
    if len(click_points) == 2:
        x1, y1 = click_points[0]
        x2, y2 = click_points[1]
        
        # Ensure the line is drawn on the annotated image
        annotated_frame_with_line = annotated_frame.copy()
        
        # Adjust the coordinates based on the resized image
        height, width, _ = annotated_frame_with_line.shape
        resized_width = width // 2
        resized_height = height // 2
        x1 = int(x1 * width / resized_width)
        y1 = int(y1 * height / resized_height)
        x2 = int(x2 * width / resized_width)
        y2 = int(y2 * height / resized_height)
        
        # Draw the line on the frame
        cv2.line(annotated_frame_with_line, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Calculate the distance between the two points
        pixel_distance = calculate_distance(x1, y1, x2, y2)
        
        # Convert pixels to inches (1 inch = 50 pixels)
        inch_distance = pixel_distance / 50
        
        # Update the label with the distance in pixels and inches
        distance_text = f"Line length: {pixel_distance:.2f} pixels, {inch_distance:.2f} inches"
        detected_label.config(text=f"{detected_label.cget('text')}\n{distance_text}")
        
        # Annotate the frame with the drawn line
        height, width, _ = annotated_frame_with_line.shape
        resized_frame = cv2.resize(annotated_frame_with_line, (width // 2, height // 2))
        
        # Convert the frame to a format suitable for Tkinter
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(img)
        
        # Update the image label
        image_label.config(image=img_tk)
        image_label.image = img_tk
        
        # Reset points list for next line
        click_points = []

# Bind mouse click event for drawing lines
image_label.bind("<Button-1>", handle_click)

# Run the Tkinter event loop
root.mainloop()
