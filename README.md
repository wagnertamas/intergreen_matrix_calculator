# SUMO Intersection Analyzer

This application allows you to analyze SUMO network files (`.net.xml`), visualize intersections, calculate intergreen matrices, and plan traffic light phase transitions.

## Features

- **Visualization**: View junction geometry, lanes, and conflict points.
- **Intergreen Matrix**: Automatically calculate intergreen times based on conflict points and approach speeds.
- **Conflict Matrix**: Define and edit conflict matrices to determine compatible signal phases.
- **Phase Planning**: Generate and visualize possible traffic light phases (cliques) based on the conflict matrix.
- **Transition Planning**: Plan detailed signal phase transitions with custom timings.
- **Export**: Export results to CSV, PDF reports, and SUMO traffic light configuration files (`.add.xml`).

## Installation

1.  **Prerequisites**:
    *   Python 3.x
    *   SUMO (Simulation of Urban MObility) installed (optional, but recommended for generating network files).

2.  **Dependencies**:
    Install the required Python packages:
    ```bash
    pip install matplotlib numpy scipy shapely networkx
    ```
    *Note: `tkinter` is usually included with Python. If you are on macOS and using Homebrew's Python, you might need to install `python-tk` or use Anaconda.*

## Usage

1.  **Run the Application**:
    Execute the `main.py` script:
    ```bash
    python main.py
    ```

2.  **Load Network**:
    A file dialog will appear. Select your SUMO network file (`.net.xml`).

3.  **Navigate Junctions**:
    Use the `<<` and `>>` buttons or the dropdown menu to switch between junctions in the network.

4.  **Analyze & Edit**:
    *   **Zoom/Pan**: Use the buttons or mouse to navigate the junction view.
    *   **Conflict Matrix**: Click on cells in the "Tiltás Mátrix" (Conflict Matrix) table to toggle conflicts manually.
    *   **Matrix Mask**: Click on cells in the "Számítás Kijelölés" table to include/exclude specific movements from calculations.

5.  **Phase Planning**:
    *   The "Lehetséges Fázisok" (Possible Phases) panel shows valid signal phases based on the current conflict matrix.
    *   Hover over a phase to highlight the compatible lanes in the visualization.

6.  **Transition Planning**:
    *   Click "Fázisátmenet Terv" (Transition Planner) to open a dialog for designing signal timing plans.
    *   Drag and drop phases to reorder them.
    *   Adjust Min Green, Yellow, and Red-Yellow times.
    *   Visualize the signal diagram.

7.  **Export**:
    *   **PDF**: Generate a visual report of the junction.
    *   **CSV**: Save the intergreen matrix to a CSV file.
    *   **SUMO Export**: Save the traffic light logic to a SUMO additional file (`.add.xml`) and a corresponding JSON file for external control (e.g., RL agents).

## File Structure

- `main.py`: Entry point of the application.
- `gui.py`: Main GUI implementation (`JunctionApp` class).
- `sumo_parser.py`: Handles parsing of SUMO `.net.xml` files.
- `geometry_utils.py`: Geometric calculations for trajectories and conflict points.
- `transition_planner.py`: Dialog for planning signal phase transitions.

## Credits

Created with the assistance of GitHub Copilot.
Based on concepts from BME Traffic Lab.
