import cv2
import torch
import re
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
from openai import OpenAI
import rclpy
from rclpy.node import Node
from std_msgs.msg import String



# Set environment variable to use X11 instead of Wayland
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Load YOLO model
yolo_model = YOLO('/home/yogesh/Downloads/results/train2/weights/best.pt')
classify_model=YOLO('/home/yogesh/aaaaaaaaaaaaaaaaa/weights/best.pt')

# Initialize Llama API client
client = OpenAI(
    api_key="LA-e7cab9edeac54ad2bc4fcfbf1b73d3939a2a8f43e1274f0c8603031a92352530",
    base_url="https://api.llama-api.com"
)

# Capture an image using the webcam
def capture_image(save_path):
    cam = None
    for index in range(4):  # Loop through possible camera indices: 0, 1, 2, 3
        cam = cv2.VideoCapture(index)
        if cam.isOpened():
            print(f"Camera successfully opened on index {index}")
            break
        cam.release()

    if cam is None or not cam.isOpened():
        print("Error: Could not open any camera.")
        return

    frame_count = 0
    saved = False

    while frame_count < 35:
        # Capture frames one by one
        ret, frame = cam.read()

        if not ret:
            print("Error: Could not read frame from camera.")
            break

        frame_count += 1
        
        # Show the video in a window
        cv2.imshow('Video Feed', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_count == 30:
            cv2.imwrite(save_path, frame)
            print(f"30th frame saved to {save_path}")
            saved = True

    if not saved:
        print("Failed to capture 30 frames.")

    # Release the camera and close windows
    cam.release()
    cv2.destroyAllWindows()

# Convert bounding box to center pixel coordinate
def get_center_of_bbox(bbox):
    x_min, y_min, x_max, y_max = bbox
    return int((x_min + x_max) / 2), int((y_min + y_max) / 2)

# Detect objects using YOLOv8 and detect colors
from PIL import Image

def detect_objects_and_colors(image_path):
    results = yolo_model(image_path)
    img = cv2.imread(image_path)
    
    detected_objects = {}  # Store object names with center pixel coordinates
    for r in results[0].boxes:
        bbox = r.xyxy[0].tolist()  # Get the bounding box
        cls = int(r.cls)
        name = results[0].names[cls].lower()  # Get class name (ball or bowl)
        
        crop = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]  # Crop the object

        if name == "ball":
            # Convert crop to PIL Image for resizing
            pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            width, height = pil_image.size
            
            # Calculate the cropping box for 36x36 crop
            left = (width - 48) / 2
            top = (height - 48) / 2
            right = (width + 48) / 2
            bottom = (height + 48) / 2
            
            # Perform center crop
            cropped_img = pil_image.crop((left, top, right, bottom))
            
            # Convert cropped image back to cv2 format and pass through YOLOv8 classifier
            cropped_img_cv2 = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
            predicted_color = yolo_classification_model(cropped_img_cv2)  # YOLOv8 classification for ball color
            
            object_name = f"{predicted_color} {name}".lower()  # e.g., 'red_ball'
        else:
            # For bowls, use detect_color function
            color = detect_color(crop).lower()  # Detect the color of the bowl
            object_name = f"{color} {name}".lower()  # e.g., 'green_bowl'

        center = get_center_of_bbox(bbox)  # Get the center pixel of the bbox
        detected_objects[object_name] = center  # Store center pixel coordinate
    
    # Print the detected objects with colors and coordinates
    print("Detected objects and their center pixel coordinates:")
    for obj_name, center in detected_objects.items():
        print(f"{obj_name}: {center}")
    
    return detected_objects


# Detect color of the object
def detect_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_ranges = {
        "red": [(0, 100, 100), (10, 255, 255), (160, 100, 100), (180, 255, 255)],
        "yellow": [(20, 100, 100), (30, 255, 255)],
        "green": [(35, 100, 100), (85, 255, 255)],
        "blue": [(100, 100, 100), (130, 255, 255)],
        "orange": [(10, 100, 100), (20, 255, 255)]  # Added orange range
    }
    color_counts = {color: 0 for color in color_ranges}
    for color, ranges in color_ranges.items():
        if color == "red":
            mask1 = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
            mask2 = cv2.inRange(hsv, np.array(ranges[2]), np.array(ranges[3]))
            mask = mask1 + mask2
        else:
            mask = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
        color_counts[color] = cv2.countNonZero(mask)
    
    detected_color = max(color_counts, key=color_counts.get)
    print(detected_color)
    if detected_color == "orange":
        return "red"
    if detected_color == "green":
        return "yellow"

    return detected_color


def yolo_classification_model(image):
    # Perform inference using YOLOv8 classification model
    results = classify_model.predict(image)

    # Get the top-1 predicted class (color label)
    predicted_class = results[0].names[results[0].probs.top1]  # Top-1 class name (predicted color)

    return predicted_class



# Send prompt to Llama API to generate actions
def generate_actions_from_prompt(prompt):
    response = client.chat.completions.create(
        model="llama3.1-405b",
        messages=[
            {"role": "system", "content": f"Simplify and break this statement and create list of actions for a robotic arm in form of put xyz ball into xyz bowl/ball/ground, \
                                         don't print anything else in output no need to mention where to pickup and what to pickup ,also if in prompt when i mention swap abc , xyz balls from positions you then those youre output should be abc ball goes to xyz ball ,xyz yellow ball goes to abc ball , give output interms of ball , bowl only and not like bin, dish, utensil, etc: {prompt}"}
        ]
    )
    lines = response.choices[0].message.content.splitlines()
    keywords = ['ball', 'bowl', 'ground', 'table']
    filtered_lines = [line for line in lines if any(keyword in line for keyword in keywords)]
    return filtered_lines

class ActionListener(Node):
    def __init__(self):
        super().__init__('action_listener')
        self.subscription = self.create_subscription(
            String,
            '/action_status',
            self.listener_callback,
            10)
        self.action_done = False

    def listener_callback(self, msg):
        if msg.data == 'action done':
            self.get_logger().info('Received: action done')
            self.action_done = True


# Execute actions and update object coordinates
# Execute actions and update object coordinates
def execute_actions(actions, detected_objects, initial_positions, img_path, node):
    color_dict = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red for arrows
    colors = ['red', 'green', 'blue', 'yellow']
    objects = ['bin', 'dish']
    destinations = ['ground', 'table', 'anywhere']
    swap_temp_storage = {}
    count = 0
    
    # Read the initial image from the img_path
    img = cv2.imread(img_path)
    
    for idx, action in enumerate(actions):
        action = action.lower()
        source_obj, destination_obj = None, None
        source_initial_center, destination_center = None, None
        count += 1
        
        # Replace 'bin' or 'dish' with 'bowl' in the action description
        for obj in objects:
            action = action.replace(f"{obj}", "bowl")
        
        ball_count = action.count("ball")
        
        # If there's only one ball, find the source and destination
        if ball_count == 1:
            # Find the source ball
            for color in colors:
                if f"{color} ball" in action:
                    source_obj = f"{color} ball"
                    source_initial_center = detected_objects.get(source_obj, None)  # Get current position from detected_objects
                    break

            # Find the destination bowl or ground
            for color in colors:
                if f"{color} bowl" in action:
                    destination_obj = f"{color} bowl"
                    destination_center = detected_objects.get(destination_obj, None)  # Get current position from detected_objects
                    break
            
            # Check if the destination is ground/table
            if any(dest in action for dest in destinations):
                destination_obj = next(dest for dest in destinations if dest in action)
                destination_center = initial_positions.get(source_obj, None)  # Use initial position only for ground/table

        # If there are two balls (e.g., "swap" or "ball-to-ball" actions)
        elif ball_count == 2:
            # Use regex to identify the order of the balls in the sentence
            pattern = r'(\w+ ball) .* (\w+ ball)'
            match = re.search(pattern, action)
            
            if match:
                # The first match is the source, the second is the destination
                source_obj = match.group(1)
                destination_obj = match.group(2)

                # For swap detection, check if this action is the reverse of the previous one
                if idx > 0 and (destination_obj, source_obj) in swap_temp_storage:
                    # Retrieve the temporarily stored coordinates for the second ball
                    destination_center = swap_temp_storage[(destination_obj, source_obj)]
                    source_initial_center = swap_temp_storage.get((source_obj, destination_obj), detected_objects[source_obj])
                else:
                    # Get the positions from detected_objects
                    source_initial_center = detected_objects.get(source_obj, None)
                    destination_center = detected_objects.get(destination_obj, None)
                    
                    # Temporarily store the source coordinates before updating
                    # swap_temp_storage[(source_obj, destination_obj)] = source_initial_center

        # Visualize action by drawing the arrow
        if source_initial_center and destination_center:
            print(f"{source_obj} from {source_initial_center} goes to {destination_obj} at {destination_center}.")
            cv2.arrowedLine(img, source_initial_center, destination_center, color_dict[idx % 3], 2, tipLength=0.05)
            
            # Display the image with the arrows drawn for 5 seconds
            cv2.imshow('Action Visualization', img)
            cv2.waitKey(10000)  # Display the image for 5000 ms (5 seconds)
            cv2.destroyWindow('Action Visualization')  # Close the window after 5 seconds
            
            # Update the detected_objects dictionary with the new position
            # detected_objects[source_obj] = destination_center
        
        else:
            print(f"Error: Coordinates not found for {source_obj} or {destination_obj}.")
        
        # Wait for "action done" signal from the real world before proceeding to the next action
        node.action_done = False
        while not node.action_done:
            rclpy.spin_once(node, timeout_sec=0.1)
            
        if node.action_done:
            print("Received 'action done' message. Proceeding with image capture and object detection.")
            
            # Capture a new image after each action and update the object coordinates
            pather = f'/home/yogesh/lo/Sapien_robo/project0/llama/images/{count}.jpg'
            capture_image(pather)
            
            # Update the image for the next action
            img = cv2.imread(pather)
            
            # Re-detect objects and update the coordinates after the new image
            detected_objects.update(detect_objects_and_colors(pather))
        else:
            print("ROS 2 node shutdown before receiving 'action done' message.")

    # Close all OpenCV windows when done
    cv2.destroyAllWindows()






def main():
    # Initialize ROS 2
    rclpy.init()
    node = ActionListener()

    # Capture initial image and detect objects
    path0='/home/yogesh/lo/Sapien_robo/project0/llama/images/0.jpg'
    capture_image(path0)
    detected_objects = detect_objects_and_colors(path0)
    initial_positions = {name: coord for name, coord in detected_objects.items() if 'ball' in name}

    # Prompt and execute actions
    try:
        while True:
            complex_prompt = input("Enter a complex prompt (or 'exit' to quit): ")
            if complex_prompt.lower() == 'exit':
                break
            
            actions = generate_actions_from_prompt(complex_prompt)
            print("Action sequences generated from the prompt:")
            for idx, action in enumerate(actions):
                print(f"Action {idx + 1}: {action}")
            
            execute_actions(actions, detected_objects, initial_positions, path0, node)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
