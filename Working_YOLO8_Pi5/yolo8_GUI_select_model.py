import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import tkinter as tk
from tkinter import Label, filedialog, StringVar, OptionMenu
from PIL import Image, ImageTk
import math
from datetime import datetime
import os

# Create the Tkinter window
root = tk.Tk()
root.title("YOLOv8 Object Detection and Line Drawing")
root.geometry(f"{1280 // 2}x{1280 // 2 + 150}")

# Initialize the camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 1280)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Model options
model_options = {
    "yolov8n.pt": YOLO("yolov8n.pt"),
    "yolov8s.pt": YOLO("yolov8s.pt"),
    "yolov8x-worldv2.pt": YOLO("yolov8x-worldv2.pt"),
    "yolov8n-obb.pt": YOLO("yolov8n-obb.pt"),
    "yolov8x-obb.pt": YOLO("yolov8x-obb.pt"),
}
selected_model = StringVar(value="yolov8n.pt")
model = model_options[selected_model.get()]



# Labels and dropdown menu for models
detected_label = Label(root, text="No objects detected.", font=("Arial", 12), fg="blue")
detected_label.pack(pady=5)

model_label = Label(root, text="Select YOLO Model:")
model_label.pack()

model_dropdown = OptionMenu(root, selected_model, *model_options.keys())
model_dropdown.pack()

# Frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(fill=tk.X, pady=5)

take_picture_button = tk.Button(button_frame, text="Take Picture", command=lambda: take_picture())
take_picture_button.pack(side=tk.LEFT, expand=True, padx=10)

import_image_button = tk.Button(button_frame, text="Import Image", command=lambda: import_image())
import_image_button.pack(side=tk.LEFT, expand=True, padx=10)

save_button = tk.Button(button_frame, text="Save Image", command=lambda: save_image())
save_button.pack(side=tk.LEFT, expand=True, padx=10)

quit_button = tk.Button(button_frame, text="Quit", command=root.quit)
quit_button.pack(side=tk.RIGHT, expand=True, padx=10)

toggle_button = tk.Button(button_frame, text="Mode: Toy Car (inches)", command=lambda: toggle_mode())
toggle_button.pack(side=tk.LEFT, expand=True, padx=10)

# Image label
image_label = Label(root)
image_label.pack()

runtime_label = Label(root, text="Run Time: N/A", font=("Arial", 12), fg="green")
runtime_label.pack(pady=5)

annotated_frame = None
click_points = []
current_mode = "toy"
unit = "inches"

def toggle_mode():
    global current_mode, unit
    if current_mode == "toy":
        current_mode = "real"
        unit = "foot"
        toggle_button.config(text="Mode: Real Car (ft)")
    else:
        current_mode = "toy"
        unit = "inch"
        toggle_button.config(text="Mode: Toy Car (inches)")
        update_detected_label()  # Update the label
        
# Function to update the detected label based on mode
def update_detected_label():
    detected_label.config(fg="green" if current_mode == "toy" else "red")


def update_model():
    global model
    model = model_options[selected_model.get()]

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_box_differences(box):
    x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
    dx = abs(x2 - x1)  # Change in x
    dy = abs(y2 - y1)  # Change in y
    diagonal = math.sqrt(dx**2 + dy**2)  # Diagonal length
    return dx, dy, diagonal

def take_picture():
    global annotated_frame
    update_model()
    start_time = datetime.now()
    frame = picam2.capture_array()
    process_frame(frame, start_time)

def import_image():
    global annotated_frame
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    if file_path and os.path.isfile(file_path):
        frame = cv2.imread(file_path)
        if frame is not None:
            start_time = datetime.now()
            process_frame(frame, start_time)
        else:
            detected_label.config(text="Invalid image selected. Please try again.")

def process_frame(frame):
    global annotated_frame, unit, normalization_factor, avg_car_phone_diagonal
    results = model(frame)
    detected_objects = []
    box_differences = []
    car_and_phone_differences = []

    if results[0].boxes:
        for box in results[0].boxes[:7]:
            class_id = int(box.cls)
            object_name = model.names[class_id]
            detected_objects.append(object_name)
            dx, dy, diagonal = calculate_box_differences(box)
            box_differences.append((dx, dy, diagonal))
            if object_name.lower() in ["car", "cell phone"]:
                car_and_phone_differences.append(diagonal)

    avg_car_phone_diagonal = (sum(car_and_phone_differences) / len(car_and_phone_differences)) if car_and_phone_differences else 0

    if detected_objects:
        object_text = f"Objects detected: {', '.join(detected_objects)}"
        differences_text = "\n".join(
            [f"{obj}: Δx={dx:.1f}, Δy={dy:.1f}, Δd={diagonal:.1f}"
             for obj, (dx, dy, diagonal) in zip(detected_objects, box_differences)]
        )
        normalization_factor = 3 if current_mode == "toy" else 15  # Inches or feet
        #normalization_factor_text = f"Pixels per Unit: {normalization_factor:.2f} "
        #pixels_per_unit = avg_car_phone_diagonal / normalization_factor
        avg_car_phone_text = f"Avg Δd (Car/Cell phone): {avg_car_phone_diagonal:.1f} pixels"
        normalization_text = f"Normalized Diagonal: {avg_car_phone_diagonal / normalization_factor:.1f} pixels per {unit}"
        #pixels_per_unit_text = f"Pixels per Unit: {pixels_per_unit:.2f} pixels/{unit}"
        #pixels_per_unit_text = f"Pixels per Unit: {pixels_per_unit:.2f} "
        #detected_label.config(text=f"{object_text}\n{differences_text}\n{avg_car_phone_text}\n{normalization_text}\n{pixels_per_unit_text}")
        factor_text = f"Normalization Factor: {normalization_factor} {unit} "
        
        detected_label.config(text=f"{object_text}\n{differences_text}\n{avg_car_phone_text}\n{normalization_text}\n{factor_text}")
    else:
        detected_label.config(text="No objects detected.")

    annotated_frame = results[0].plot()
    update_image_label(annotated_frame)
    


def update_image_label(frame):
    height, width, _ = frame.shape
    resized_frame = cv2.resize(frame, (width // 2, height // 2))
    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

def save_image():
    if annotated_frame is not None:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{current_time}.png"
        cv2.imwrite(filename, annotated_frame)
        print(f"Image saved as {filename}")

def handle_click(event):
    global click_points, unit, normalization_factor, avg_car_phone_diagonal
    click_points.append((event.x, event.y))
    if len(click_points) == 2:
        x1, y1 = click_points[0]
        x2, y2 = click_points[1]
        annotated_frame_with_line = annotated_frame.copy()
        height, width, _ = annotated_frame_with_line.shape
        x1 = int(x1 * width / (width // 2))
        y1 = int(y1 * height / (height // 2))
        x2 = int(x2 * width / (width // 2))
        y2 = int(y2 * height / (height // 2))
        cv2.line(annotated_frame_with_line, (x1, y1), (x2, y2), (0, 255, 255), 8)
        pixel_distance = calculate_distance(x1, y1, x2, y2)
        #scaled_distance = pixel_distance / normalization_factor
        scaled_distance = pixel_distance * normalization_factor / avg_car_phone_diagonal 
        
        distance_text = f"Line length: {pixel_distance:.2f} pixels, {scaled_distance:.2f} {unit}"
        detected_label.config(text=f"{detected_label.cget('text')}\n{distance_text}")
  
        update_image_label(annotated_frame_with_line)
        click_points = []

image_label.bind("<Button-1>", handle_click)
root.mainloop()
