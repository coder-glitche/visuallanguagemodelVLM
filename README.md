

---

# Robotic Arm Action Execution using YOLOv8 and Llama API

This project demonstrates how to use a YOLOv8 model for object detection and OpenAI's Llama API for generating robotic arm actions from complex language prompts. The detected objects (balls and bowls) are then used to execute actions based on prompts, such as moving a ball into a bowl or swapping balls. Additionally, the system integrates ROS 2 to listen for feedback from the real world and capture new images after each action.

## Prerequisites

## Setup

### Environment Setup

1. Clone the repository or place your scripts in the project directory:
    ```bash
    git clone https://github.com/yourusername/robot-arm-control.git
    cd robot-arm-control
    ```

2. Install the Python dependencies:
    ```bash
    pip install opencv-python-headless numpy torch ultralytics rclpy openai
    ```

### Llama API Key

1. Create an account and get an API key from Llama API: https://api.llama-api.com
2. Replace the following placeholder in the script with your actual Llama API key:
    ```python
    client = OpenAI(api_key="your_api_key_here", base_url="https://api.llama-api.com")
    ```

### ROS 2 Setup

1. Install ROS 2 Humble as described in the [official documentation](https://docs.ros.org/en/humble/Installation.html).
2. Ensure ROS 2 environment is correctly sourced:
    ```bash
    source /opt/ros/humble/setup.bash
    ```

### YOLOv8 Model

1. You need a trained YOLOv8 model for detecting colored balls and bowls find it [here](). 
2. Place the trained YOLO model in the desired directory and update the path in the script:
    ```python
    yolo_model = YOLO('/path/to/your/yolo/best.pt')
    ```
3. To train a custom model execute the following in your command line terminal
    ```python
     yolo detect train data='/path/to/data.yaml' model=yolov8n.pt epochs=100
    ```
    refer this [documentation](https://docs.ultralytics.com/usage/cli/) for all training and testing options


## How It Works

1. **Object Detection**: YOLOv8 detects objects (balls and bowls) and identifies their colors.
2. **Action Generation**: Llama API processes a complex prompt and generates simplified action sequences.
3. **ROS 2 Integration**: A listener node waits for a signal indicating that the real-world action is complete.
4. **Image Capture and Update**: After each action, a new image is captured, and the objects' positions are updated.

---

## Description of Functions

### 1. `capture_image(save_path)`
**Purpose**: Captures an image from the webcam and saves it to the specified path.

- **Input**: `save_path` (string) – The file path where the captured image will be saved.
- **Output**: Saves the 30th frame captured from the webcam as an image file.
- **Process**: 
  - Loops through possible camera indices (0, 1, 2, 3) to find a working camera.
  - Captures video frames until 30 frames are reached and saves the 30th frame.
  - Displays a video feed in a window until 'q' is pressed.
  
### 2. `get_center_of_bbox(bbox)`
**Purpose**: Calculates the center pixel coordinates of a bounding box.

- **Input**: `bbox` (list of four floats) – The bounding box coordinates `[x_min, y_min, x_max, y_max]` from YOLOv8.
- **Output**: Returns the center pixel coordinates as `(x_center, y_center)`.

### 3. `detect_objects_and_colors(image_path)`
**Purpose**: Detects objects and their colors in the provided image using the YOLO model and color detection.

- **Input**: `image_path` (string) – Path to the image in which objects need to be detected.
- **Output**: A dictionary of detected objects with their center pixel coordinates, e.g., `{ 'red ball': (x, y), 'green bowl': (x, y) }`.
- **Process**: 
  - YOLOv8 detects the objects in the image.
  - Each detected object is cropped, and its color is detected using HSV color space.
  - The center of the bounding box is calculated for each object.

### 4. `detect_color(image)`
**Purpose**: Detects the dominant color in the cropped image of an object.

- **Input**: `image` (cropped image of the object in BGR format).
- **Output**: Returns the name of the dominant color (e.g., 'red', 'blue', 'green', or 'yellow').
- **Process**: 
  - Converts the image to HSV color space.
  - Applies a mask to determine the number of pixels corresponding to each color.
  - The color with the highest count is selected as the object's color.

### 5. `generate_actions_from_prompt(prompt)`
**Purpose**: Uses the Llama API to generate simplified action sequences for the robotic arm from a natural language prompt.

- **Input**: `prompt` (string) – A complex natural language prompt describing the desired actions.
- **Output**: A list of actions in the form of `["red ball goes to green bowl", "blue ball goes to ground"]`.
- **Process**: 
  - The Llama API processes the prompt and simplifies it into structured actions.
  - The function filters lines that contain keywords like 'ball', 'bowl', 'ground', or 'table' to generate relevant actions.

### 6. `ActionListener(Node)`
**Purpose**: A ROS 2 listener node that waits for a message indicating that the real-world action is complete.

- **Input**: Inherits from `Node`.
- **Output**: Sets a flag when it receives the message 'action finished in real world'.
- **Process**: 
  - Subscribes to the `action_status` topic.
  - The callback function sets the `received` attribute to `True` when the expected message is heard.

### 7. `execute_actions(actions, detected_objects, initial_positions, img, node)`
**Purpose**: Executes the sequence of actions generated from the prompt and updates the coordinates of the objects after each action.

- **Input**:
  - `actions` (list) – List of actions generated from the prompt.
  - `detected_objects` (dict) – Dictionary of detected objects with their pixel coordinates.
  - `initial_positions` (dict) – Initial coordinates of the detected objects.
  - `img` (image) – The current image where arrows will be drawn to show object movement.
  - `node` (ROS 2 Node) – The ROS 2 node to listen for action completion.
- **Output**: Visualizes the movement of objects and updates their coordinates in the `detected_objects` dictionary.
- **Process**:
  - For each action, it identifies the source and destination objects (ball, bowl, ground).
  - Draws an arrow from the source to the destination.
  - Updates the `detected_objects` dictionary with the new positions after each action.
  - Waits for the ROS 2 listener node to confirm that the action has been completed in the real world before proceeding to the next action.
  - Captures a new image after each action and updates object coordinates.

### 8. `Talker(Node)`
**Purpose**: A ROS 2 talker node that periodically publishes a message to indicate that an action has been completed in the real world.

- **Input**: Inherits from `Node`.
- **Output**: Publishes a message to the `action_status` topic every 20 seconds.
- **Process**: 
  - Creates a publisher on the `action_status` topic.
  - Publishes the message 'action finished in real world' to signal the completion of an action.

---

## Running the Project

1. **Start the ROS 2 Talker Node**: 
   - Run the `talker.py` script to start the talker node, which will publish a message after each action.
   
   ```bash
   python3 talker.py
   ```

2. **Run the Main Script**:
   - Run the main script that captures images, detects objects, and executes actions.

   ```bash
   python3 rose.py
   ```

3. **Provide Prompts**:
   - When prompted in the console, provide a natural language prompt describing the actions you want the robotic arm to perform.

---

## Notes

- Ensure that the YOLO model is properly trained and loaded with the correct path in the `yolo_model = YOLO(...)` line.
- The image capture and ROS 2 setup must be configured properly on your system for the code to run as expected.

---
