import tkinter as tk
from PIL import Image, ImageTk
import math

class LineDrawingApp:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Draw Lines on Image")
        
        # Set window size
        root.geometry("1300x800")  # Slightly larger than the image for padding
        
        # Load and resize the image
        self.image = Image.open(image_path)
        self.image = self.image.resize((1280, 720), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.image)
        
        # Create a canvas
        self.canvas = tk.Canvas(root, width=1280, height=720)
        self.canvas.pack()
        
        # Display the resized image on the canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        # Draw a 50x50 square to represent 1 inch
        self.canvas.create_rectangle(20, 20, 70, 70, outline="blue", width=2)
        self.canvas.create_text(45, 75, text="1 inch", fill="blue", font=("Arial", 10))
        
        # Line drawing variables
        self.points = []  # List to store the two points
        
        # Label to display the distance
        self.distance_label = tk.Label(root, text="Distance: 0.00 inches", font=("Arial", 14))
        self.distance_label.pack(pady=10)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.handle_click)
        
        # Add Quit button
        quit_button = tk.Button(root, text="Quit", command=root.destroy)
        quit_button.pack(pady=10)
        
        # Bind Escape key to close
        root.bind("<Escape>", lambda event: root.destroy())
    
    def handle_click(self, event):
        # Store the clicked point
        self.points.append((event.x, event.y))
        
        # If two points are clicked, draw a line
        if len(self.points) == 2:
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]
            self.canvas.create_line(x1, y1, x2, y2, fill="red", width=2)
            
            # Calculate the distance in pixels
            pixel_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
            # Convert pixels to inches (1 inch = 50 pixels)
            inch_distance = pixel_distance / 50
            
            # Update the label with the distance
            self.distance_label.config(text=f"Distance: {inch_distance:.2f} inches")
            
            # Reset points list for next line
            self.points = []

# Main execution
if __name__ == "__main__":
    # Path to your image
    image_path = "your_image.jpg"  # Replace with your image file path
    
    # Create the main window
    root = tk.Tk()
    app = LineDrawingApp(root, image_path)
    root.mainloop()
