#!/usr/bin/env python3
import sys
import os

# ensure project root is importable and run gui.viewer
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from PySide6.QtWidgets import QApplication
from gui.viewer import ImageViewer


def main():
    app = QApplication(sys.argv)
    win = ImageViewer()
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
