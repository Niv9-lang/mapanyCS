# MapAnything — Setup and Usage Guide

MapAnything reconstructs a physical space in 3D from photographs, then lets you explore it, measure distances, and plan obstacle-avoiding paths.

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Installation](#2-installation)
3. [3D Reconstruction](#3-3d-reconstruction)
4. [3D Explorer (PLY Explorer)](#4-3d-explorer-ply-explorer)
5. [Navigation and Pathfinding (Flask)](#5-navigation-and-pathfinding-flask)
6. [Managing Multiple Spaces](#6-managing-multiple-spaces)
7. [Benchmark](#7-benchmark)

---

## 1. Requirements

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| GPU | CPU only (slow) | NVIDIA (CUDA) or Apple Silicon (MPS) |
| Python | 3.10 | 3.12 |

> Without a GPU, 3D reconstruction still works but may take several minutes.

### Software

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (recommended for environment management)
- A modern web browser (Chrome, Firefox, Safari)

---

## 2. Installation

### Step 1 — Create the Python environment

```bash
conda create -n mapanything python=3.12 -y
conda activate mapanything
```

### Step 2 — Install PyTorch

**NVIDIA GPU:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Apple Silicon or CPU:**
```bash
pip install torch torchvision torchaudio
```

### Step 3 — Install MapAnything

From the project root:
```bash
pip install -e .
```

### Step 4 — Install additional dependencies

```bash
pip install open3d flask
```

| Package | Purpose |
|---|---|
| `open3d` | Required for 3D reconstruction |
| `flask` | Required for the web navigation tools |

### Optional dependencies (benchmark only)

```bash
pip install psutil matplotlib pynvml
```

---

## 3. 3D Reconstruction

`reconstruction_medium.py` takes a folder of photographs of a space and produces a `.ply` file (a coloured 3D point cloud).

### Usage

```bash
python reconstruction_medium.py \
    --image_folder <photo_folder>/ \
    --output <name>.ply
```

**Example:**
```bash
python reconstruction_medium.py \
    --image_folder img_room_101/ \
    --output room_101.ply
```

### Options

| Option | Description |
|---|---|
| `--image_folder` | Folder containing the photos (JPG or PNG) |
| `--output` | Output filename (`.ply`) |
| `--no_visualize` | Skip the preview window after reconstruction |

### Output files

- **`<name>.ply`** — the 3D reconstruction; place it in `database/` to use it with PLY Explorer
- **`<photo_folder>/poses.json`** — camera positions (generated automatically)

> Models are downloaded automatically from HuggingFace on first run.

---

## 4. 3D Explorer (PLY Explorer)

`PLY_explorer.html` lets you freely navigate a 3D reconstruction directly in the browser, with no additional setup.

### Launch

```bash
python -m http.server 8080
```

Then open in your browser: **`http://localhost:8080/PLY_explorer.html`**

You can also drag and drop a `.ply` file directly into the browser window.

---

### Keyboard Shortcuts

#### Movement

| Key | Action |
|---|---|
| `Z` or `W` | Move forward |
| `S` | Move backward |
| `Q` or `A` | Move left |
| `D` | Move right |
| `Space` | Jump / Move up (fly mode) |
| `Shift` | Sprint / Move down (fly mode) |
| `N` | Toggle **fly mode** (move freely through walls) |

> ZQSD keys correspond to AZERTY keyboards. WASD also works (QWERTY).

#### Interface

| Key | Action |
|---|---|
| `M` | Expand / collapse the minimap |
| `G` | Cycle the **vertical axis** if the model appears upside down or on its side |
| `C` | Clear the current path |
| `L` | Open the **space transition menu** (if configured) |
| `Escape` | Open the pause menu / release the mouse |

---

### Features

#### Immersive navigation
First-person view with a captured mouse pointer. The camera follows mouse movement to look around. Click on the scene after pressing `Escape` to resume navigation.

#### Minimap
A real-time 2D map displayed in the top-right corner. Shows the player's position, annotated rooms, and transition points.

**When pressing `M` (expanded minimap):**
- The occupancy grid is shown in colour:
  - **Dark blue** = free navigable area
  - **Red** = obstacle
- **1st click** on the map → sets start point A
- **2nd click** → sets end point B and automatically computes the path
- **3rd click** → resets both points
- The result is shown at the bottom: distance in metres and number of waypoints
- The **Floor±** slider adjusts the obstacle detection sensitivity (default: 20 cm — objects less than this height above the floor are ignored as obstacles)

#### Annotated rooms and JSON configuration

PLY Explorer can automatically display room labels, proximity notifications, and transition markers when a **JSON annotation file** is provided alongside the reconstruction.

**How it works:**
- The JSON file must be placed in the **same folder** as its corresponding `.ply` file
- It must have the **exact same filename**, only the extension differs

```
database/
├── vivant_RDC.ply       ← 3D reconstruction
└── vivant_RDC.json      ← annotation file (same name, same folder)
```

If PLY Explorer finds a matching JSON file when a reconstruction is loaded, it reads it automatically — no manual action required.

**What the JSON file enables:**
- **Room labels** — named locations displayed as floating labels in the 3D scene and on the minimap. A notification appears automatically when the player moves within range of a room.
- **Space transitions** — golden markers placed at doorways, staircases, or elevators. Pressing `L` near a marker opens a menu to travel to another reconstruction.
- **Floor calibration** — the `floorY` value overrides the automatic floor detection, ensuring the player spawns at the correct height and the occupancy grid is computed accurately.
- **Vertical axis** — the `upPreset` value tells PLY Explorer which axis is "up" in this particular file, so the model is oriented correctly on load.

**Minimal JSON example** (rooms only, no transitions):
```json
{
  "proximite_seuil": 2.0,
  "floorY": -0.85,
  "upPreset": 1,
  "salles": [
    {
      "id": "101",
      "nom": "Room 101",
      "x": 1.23,
      "y": 0.0,
      "z": -2.45,
      "description": "Classroom — Ground floor"
    }
  ]
}
```

| Field | Description |
|---|---|
| `proximite_seuil` | Distance (in scene units) within which a room label becomes active |
| `floorY` | Floor height in viewer coordinates — adjust if the player spawns above or below the floor |
| `upPreset` | Vertical axis — `1` for MASt3r / MapAnything outputs, `2` for CloudCompare exports |
| `salles` | Array of rooms, each with an `id`, display name (`nom`), XYZ position, and optional `description` |

> The XYZ coordinates for rooms and transitions correspond to positions as seen in the viewer's HUD (displayed bottom-left while navigating).

#### Space transitions
Golden markers indicate passage points to other spaces (stairs, elevators, etc.). Approach a marker and press `L` to choose your destination. See [Section 6](#6-managing-multiple-spaces) for the full transition format including arrival coordinates (`spawn`).

#### Orientation correction
If the model appears upside down or lying on its side, press `G` to cycle through available orientations until the vertical axis looks correct. Once the correct orientation is found, save it as `upPreset` in the JSON file so it is applied automatically on future loads.

---

## 5. Navigation and Pathfinding (Flask)

`Navigation/navigation_obstacle.py` starts a local web server with two tools that complement PLY Explorer.

### Important prerequisite

Open PLY Explorer and load the reconstruction **before** using these pages — the navigation grid is transmitted automatically.

### Launch

```bash
python Navigation/navigation_obstacle.py --model database/<name>.ply
```

**Example:**
```bash
python Navigation/navigation_obstacle.py --model database/room_101.ply
```

### Options

| Option | Description |
|---|---|
| `--model` | `.ply` file to load |
| `--salles` | Annotation JSON file (default: same name as the `.ply`) |
| `--host` | Listening address (default: `127.0.0.1`) |
| `--port` | Port (default: `5000`) |

### Available pages

#### `http://127.0.0.1:5000/pathfinding` — A→B Pathfinding

Displays the occupancy grid sent by PLY Explorer.

- **Click** on the map to place start point A, then end point B
- The path is computed automatically, avoiding obstacles
- Distance is shown in metres
- A 3rd click clears both points
- The **Floor tolerance** slider sets the minimum height above the floor at which an object is considered an obstacle (default: 20 cm)

#### `http://127.0.0.1:5000/minimap-distance` — Distance measurement

- **Click** two points on the map to measure the straight-line distance between them
- Result displayed in metres

---

## 6. Managing Multiple Spaces

To access multiple reconstructions from PLY Explorer at startup:

### File structure

```
map-anything/
└── database/
    ├── index.json          ← list of available spaces
    ├── room_101.ply
    ├── room_101.json       ← annotations (rooms, transitions, floor)
    ├── hallway_GF.ply
    └── hallway_GF.json
```

> The `.json` file must have exactly the same name as its corresponding `.ply` file.

### `database/index.json`

```json
[
  {
    "file": "room_101",
    "name": "Room 101"
  },
  {
    "file": "hallway_GF",
    "name": "Ground floor hallway"
  }
]
```

### `<space>.json` — Annotation file

```json
{
  "proximite_seuil": 2.0,
  "floorY": -0.85,
  "upPreset": 1,
  "salles": [
    {
      "id": "101",
      "nom": "Room 101",
      "x": 1.23,
      "y": 0.0,
      "z": -2.45,
      "description": "Classroom"
    }
  ],
  "transitions": [
    {
      "id": "T1",
      "nom": "Elevator",
      "x": 4.5,
      "y": 0.0,
      "z": 1.2,
      "destinations": [
        {
          "file": "hallway_GF",
          "nom": "Ground floor hallway"
        },
        {
          "file": "room_101",
          "nom": "Room 101",
          "spawn": { "x": 1.2, "y": -0.85, "z": 3.0 }
        }
      ]
    }
  ]
}
```

| Field | Description |
|---|---|
| `proximite_seuil` | Distance (in scene units) at which a room is flagged as nearby |
| `floorY` | Floor height — adjust if automatic detection is incorrect |
| `upPreset` | Vertical axis orientation — `1` for MASt3r/MapAnything, `2` for CloudCompare |
| `salles` | List of rooms with their XYZ position in the scene |
| `transitions` | Passage points to other PLY files. Each transition can have multiple `destinations`. The optional `spawn` field sets the XYZ arrival coordinates in the destination space — if omitted, the player spawns at the centre of the target space. |

---

## 7. Benchmark

`benchmark.py` measures the time and resources required to reconstruct a space as a function of the number of photos used.

### Launch

```bash
python benchmark.py --image_folder img_room_101/
```

The machine name is requested at startup to identify results.

### Compare results across machines

```bash
python benchmark.py --compare
```

Results are saved to `benchmark_results/all_results.json`.

---

## Full workflow

```
1.  Take photos of a space

2.  python reconstruction_medium.py
        --image_folder <photos>/
        --output database/<space>.ply

3.  Place database/<space>.ply and create database/<space>.json

4a. PLY Explorer  →  python -m http.server 8080
                     http://localhost:8080/PLY_explorer.html

4b. Pathfinding   →  python Navigation/navigation_obstacle.py --model database/<space>.ply
                     http://127.0.0.1:5000/pathfinding
```
