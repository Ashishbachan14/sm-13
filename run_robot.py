import cv2
import numpy as np
import json
import heapq
import os
import time

# --- Mock Robot API ---
# Since the 'robot_api' module is not available, this mock class
# simulates the robot's behavior by printing its actions.
class RobotController:
    """
    A mock robot controller for demonstration purposes.
    This class simulates the robot's movements by printing actions to the console.
    """
    def __init__(self):
        print("Robot controller initialized.")

    def move_to(self, x, z, wait_time=0.1):
        """Simulates moving to a specific waypoint."""
        print(f"  -> Moving to waypoint: (x={x:.2f}, z={z:.2f})")
        # In a real scenario, this would command the robot and might wait for completion.
        time.sleep(wait_time)

    def stop(self):
        """Simulates stopping the robot."""
        print("Robot stopped.")


# ==============================================================================
# --- PART 1: OBSTACLE DETECTION ---
# ==============================================================================

# --- Configuration Constants for Obstacle Detection ---
WORLD_GRID_DIMENSION = 100.0  # The dimensions of your simulation's coordinate grid.
IMAGE_DIMENSION = 1920.0      # The dimensions of the input image file in pixels.

def find_active_area_width(image):
    """
    Automatically detects the main light gray simulation area to get its
    precise pixel width for accurate scaling.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        float: The width of the detected active area in pixels, or None if not found.
    """
    # Convert to grayscale for easier thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to isolate the light gray area (value is ~211) from the dark border (~54)
    # We choose a value in between, like 150.
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Find all contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    # Find the largest contour, which should be the active area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding box of the largest contour
    _x, _y, w, _h = cv2.boundingRect(largest_contour)
    
    return float(w)


def extract_obstacle_data(image_path, output_json_path="obstacles.json"):
    """
    Analyzes an image to find green obstacles, automatically calibrates the scale,
    calculates world coordinates, and saves the data to a JSON file.

    Args:
        image_path (str): The file path to the input image.
        output_json_path (str): The path to save the output JSON file.
        
    Returns:
        bool: True if extraction was successful, False otherwise.
    """
    # 1. Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return False

    # 2. Automatically find the active area width for scaling
    active_area_width = find_active_area_width(image)
    if active_area_width is None:
        print("Error: Could not detect the active simulation area. Aborting.")
        return False
        
    print(f"  Auto-detected active area width: {active_area_width} pixels.")

    # 3. Define scale factor and center
    scale_factor = WORLD_GRID_DIMENSION / active_area_width
    pixel_center = IMAGE_DIMENSION / 2.0

    # 4. Perform color segmentation for green obstacles
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 5. Find contours of the obstacles
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obstacles_data = []

    # 6. Process each obstacle contour
    for cnt in contours:
        if cv2.contourArea(cnt) < 20:  # Filter noise
            continue

        rect = cv2.minAreaRect(cnt)
        (px, pz_pixel), (width_px, height_px), _ = rect

        # 7. Transform coordinates using the new dynamic scale factor
        world_x = (px - pixel_center) * scale_factor
        world_z = (pz_pixel - pixel_center) * scale_factor

        world_width = width_px * scale_factor
        world_height = height_px * scale_factor

        obstacle = {
            "center": {
                "x": round(world_x, 4),
                "z": round(world_z, 4)
            },
            "size": {
                "width": round(world_width, 4),
                "height": round(world_height, 4)
            }
        }
        obstacles_data.append(obstacle)

    # 8. Write the data to a JSON file
    try:
        with open(output_json_path, 'w') as f:
            json.dump(obstacles_data, f, indent=4)
        print(f" Success! Detected {len(obstacles_data)} obstacles.")
        print(f"   Data saved to '{output_json_path}'")
        return True
    except IOError as e:
        print(f"Error: Could not write to file {output_json_path}. Reason: {e}")
        return False


# ==============================================================================
# --- PART 2: A* PATH PLANNING ---
# ==============================================================================

# --- Configuration for Path Planning ---
ROBOT_RADIUS = 2.5    # Safety margin. Increase if the robot is clipping obstacles.
GRID_RESOLUTION = 1.0 # The distance between nodes in our search grid.

class Grid:
    """
    Represents the simulation environment, including obstacles.
    """
    def __init__(self, obstacles_data, safety_radius):
        self.obstacles = obstacles_data
        self.safety_radius = safety_radius
        print(f" Grid initialized with {len(self.obstacles)} obstacles.")

    def is_colliding(self, x, z):
        """
        Checks if a given coordinate is inside any obstacle, including a safety margin.

        Args:
            x (float): The x-coordinate to check.
            z (float): The z-coordinate to check.

        Returns:
            bool: True if the coordinate is in collision, False otherwise.
        """
        for obs in self.obstacles:
            center_x = obs['center']['x']
            center_z = obs['center']['z']
            size_x = obs['size']['width']
            size_z = obs['size']['height']
            
            # Calculate the expanded boundaries for collision checking
            min_x = center_x - (size_x / 2) - self.safety_radius
            max_x = center_x + (size_x / 2) + self.safety_radius
            min_z = center_z - (size_z / 2) - self.safety_radius
            max_z = center_z + (size_z / 2) + self.safety_radius

            if min_x <= x <= max_x and min_z <= z <= max_z:
                return True # Collision detected
        return False


def a_star_search(start, goal, grid):
    """
    Finds the shortest path from start to goal using the A* algorithm.

    Args:
        start (tuple): The starting (x, z) coordinates.
        goal (tuple): The goal (x, z) coordinates.
        grid (Grid): The grid object representing the environment.

    Returns:
        list: A list of (x, z) tuples representing the path, or None if no path is found.
    """
    start_node = (start[0], start[1])
    goal_node = (goal[0], goal[1])

    open_set = [(0, start_node)]  # (f_score, node)
    came_from = {}
    g_score = {start_node: 0}

    def heuristic(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    while open_set:
        _current_f_score, current_node = heapq.heappop(open_set)

        if np.linalg.norm(np.array(current_node) - np.array(goal_node)) < GRID_RESOLUTION:
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start_node)
            return path[::-1] # Return reversed path

        # Explore neighbors (8 directions)
        for dx in [-GRID_RESOLUTION, 0, GRID_RESOLUTION]:
            for dz in [-GRID_RESOLUTION, 0, GRID_RESOLUTION]:
                if dx == 0 and dz == 0:
                    continue
                
                neighbor = (current_node[0] + dx, current_node[1] + dz)
                
                if grid.is_colliding(neighbor[0], neighbor[1]):
                    continue

                tentative_g_score = g_score[current_node] + np.sqrt(dx**2 + dz**2)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score, neighbor))

    return None # No path found


# ==============================================================================
# --- MAIN EXECUTION SCRIPT ---
# ==============================================================================

if __name__ == "__main__":
    # --- Define File Paths and Goal ---
    image_file = "robot_map_topview_1758948977371.png"
    obstacles_file = "obstacles.json"
    
    start_pos = (0.0, 0.0)
    goal_pos = (45.0, -45.0)

    print("--- Running Step 1: Obstacle Detection ---")
    if not os.path.exists(image_file):
        print(f" Critical Error: Image file not found at '{image_file}'")
        print("   Please make sure the image is in the same directory as the script.")
    elif extract_obstacle_data(image_file, obstacles_file):
        print("\n--- Running Step 2: Path Planning and Execution ---")
        
        # 1. Load Obstacle Data
        try:
            with open(obstacles_file, 'r') as f:
                obstacles = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f" Error loading obstacles file: {e}")
            exit()

        # 2. Initialize Environment and Robot Controller
        grid = Grid(obstacles, ROBOT_RADIUS)
        controller = RobotController()

        # 3. Find Path
        print(f"\nSeeking path from {start_pos} to {goal_pos}...")
        path = a_star_search(start_pos, goal_pos, grid)

        # 4. Execute Path
        if path:
            print(f" Path found with {len(path)} waypoints. Executing...")
            for waypoint in path:
                controller.move_to(waypoint[0], waypoint[1])
            # Final move to the exact goal position for precision
            controller.move_to(goal_pos[0], goal_pos[1], wait_time=0)
            print("\n Goal Reached!")
        else:
            print(" No path could be found to the goal.")

        controller.stop()
    else:
        print("\n Obstacle detection failed. Halting execution.")
