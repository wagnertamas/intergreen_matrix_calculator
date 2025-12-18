import tkinter as tk
from tkinter import filedialog, messagebox
import sys
import os

from sumo_parser import SumoInternalParser
from gui import JunctionApp

def main():
    """
    The main entry point for the SUMO Intersection Analyzer application.
    
    This function:
    1. Initializes the Tkinter root window.
    2. Prompts the user to select a SUMO network file (.net.xml).
    3. Parses the network file using SumoInternalParser.
    4. Launches the JunctionApp GUI.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window initially

    # Prompt user to select a file
    file_path = filedialog.askopenfilename(
        title="Select SUMO Network File",
        filetypes=[("SUMO Network Files", "*.net.xml"), ("XML Files", "*.xml"), ("All Files", "*.*")]
    )

    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(0)

    try:
        # Parse the selected file
        parser = SumoInternalParser(file_path)
        
        # Show the main window and start the application
        root.deiconify()
        app = JunctionApp(root, parser)
        root.mainloop()
        
    except Exception as e:
        # Handle any errors during initialization
        messagebox.showerror("Error", f"Failed to load network file:\n{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
