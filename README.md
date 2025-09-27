# 🤖 Obstacle Robot Game — SM13 

## 📌 Overview
This project implements a **3D obstacle-avoidance robot game** with three main components:

1. **Web Simulator (`index.html`)** → A browser-based 3D environment (Three.js) showing the robot, obstacles, and goal.  
2. **Server (`server.py`)** → Flask + WebSocket API that connects the simulator to external clients.  
3. **Robot Client (`runrobot.py`)** → Detects obstacles, runs A* path planning, and sends movement commands to the robot.  

The setup allows experimenting with different strategies:
- **Accuracy-first** → prioritize avoiding collisions  
- **Speed-first** → reach goal quickly, allowing some collisions  
---

## 🔹 Components

### 1. `server.py`
- Runs a **Flask REST API** (port `5000`) and a **WebSocket server** (port `8080`)
- Broadcasts control commands to connected simulators
- Supports:
  - `/move` → Move robot to absolute position  
  - `/move_rel` → Move robot relative to its current position  
  - `/stop` → Stop robot  
  - `/capture` and `/capture_top_view` → Request camera snapshots  
  - `/goal` → Set goal position (via corners or coordinates)  
  - `/obstacles/positions` → Place obstacles  
  - `/obstacles/motion` → Enable or disable obstacle movement  
  - `/collisions` → Query number of collisions recorded  
  - `/reset` → Reset environment and collision count  

### 2. `runrobot.py`
- **Part 1: Obstacle Detection**
  - Loads a top-view map image (default: `robot_map_topview_1758948977371.png`)
  - Uses OpenCV color segmentation to detect **green obstacles**
  - Calibrates pixel → world scaling automatically
  - Saves results into `obstacles.json`

- **Part 2: Path Planning**
  - Implements **A\* search** on a grid with safety margin
  - Respects robot radius during collision checks
  - Returns a waypoint path from **start** → **goal**

- **Part 3: Execution**
  - Uses `RobotController` (mock) to “move” along waypoints
  - Prints each move to console
  - Ends when robot reaches the goal

---

## ⚖️ Accuracy vs Speed
- **Accuracy-first** → Larger safety radius, avoids all collisions  
- **Speed-first** → Smaller safety margin, shorter path but more risk  

---

## 🛠️ Tech Stack
- Python 3.11+  
- Flask + WebSockets (server)  
- OpenCV (obstacle detection)  
- NumPy (math, grid handling)  
- heapq (A\* priority queue)  

---

## ▶️ How to Run

### 1. Start the Server
```bash
pip install -r requirements.txt
python run_robot.py
index.html
