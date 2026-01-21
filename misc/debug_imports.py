import sys
print("Starting import check...")

try:
    print("Importing numpy...")
    import numpy
    print(f"numpy version: {numpy.__version__}")
except Exception as e:
    print(f"numpy failed: {e}")

try:
    print("Importing matplotlib...")
    import matplotlib
    print(f"matplotlib version: {matplotlib.__version__}")
except Exception as e:
    print(f"matplotlib failed: {e}")

try:
    print("Importing torch...")
    import torch
    print(f"torch version: {torch.__version__}")
except Exception as e:
    print(f"torch failed: {e}")

try:
    print("Importing shapely...")
    from shapely.geometry import Point
    print("shapely imported.")
except Exception as e:
    print(f"shapely failed: {e}")

try:
    print("Importing scipy...")
    import scipy.interpolate
    print(f"scipy version: {scipy.__version__}")
except Exception as e:
    print(f"scipy failed: {e}")

try:
    print("Importing networkx...")
    import networkx
    print(f"networkx version: {networkx.__version__}")
except Exception as e:
    print(f"networkx failed: {e}")

except Exception as e:
    print(f"networkx failed: {e}")

try:
    print("Importing tkinter...")
    import tkinter as tk
    root = tk.Tk()
    print("Tkinter initialized.")
except Exception as e:
    print(f"Tkinter failed: {e}")

except Exception as e:
    print(f"Tkinter failed: {e}")

try:
    print("Importing tkinter.ttk...")
    from tkinter import ttk
    s = ttk.Style()
    print("ttk imported.")
except Exception as e:
    print(f"ttk failed: {e}")

try:
    print("Importing matplotlib.pyplot & backend...")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    print("matplotlib backend imported.")
except Exception as e:
    print(f"matplotlib backend failed: {e}")

try:
    print("Importing libsumo...")
    # Add SUMO_HOME/tools to path if needed, but assuming PYTHONPATH is set as before
    import libsumo
    print(f"libsumo version: {libsumo.version.get()}")
except Exception as e:
    print(f"libsumo failed: {e}")
except ImportError:
    print("libsumo not found (ImportError)")

try:
    if 'root' in locals():
        print("Updating root window...")
        root.update()
        print("Root window updated.")
except Exception as e:
    print(f"Root update failed: {e}")

print("Import check finished.")
