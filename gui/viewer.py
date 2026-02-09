from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QRadioButton,
    QButtonGroup,
    QHBoxLayout,
    QSlider,
    QScrollArea,
    QFileDialog,
)
from PySide6.QtGui import QPixmap, QColor, QImage
from PySide6.QtCore import Qt
import numpy as np

from dahu import (
    immersion,
    level_lines_distance_transform,
    get_coordinates,
    get_marker_image,
)
from .imagelabel import ImageLabel
from .results_window import ResultsWindow


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer")
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.open_btn = QPushButton("Open Image")
        self.open_btn.clicked.connect(self.open_image)
        layout.addWidget(self.open_btn)

        # Clear button (disabled until an image is loaded)
        self.clear_btn = QPushButton("Clear Painting")
        self.clear_btn.setEnabled(False)
        self.clear_btn.clicked.connect(self._on_clear)
        layout.addWidget(self.clear_btn)

        # Computation button
        self.compute_btn = QPushButton("Compute")
        self.compute_btn.clicked.connect(self._on_compute)
        layout.addWidget(self.compute_btn)

        # Brush controls (radio buttons)
        hbox = QHBoxLayout()
        self.fg_radio = QRadioButton("Foreground (Blue)")
        self.bg_radio = QRadioButton("Background (Red)")
        self.brush_group = QButtonGroup(self)
        self.brush_group.addButton(self.fg_radio, 0)
        self.brush_group.addButton(self.bg_radio, 1)
        self.fg_radio.setChecked(True)
        hbox.addWidget(self.fg_radio)
        hbox.addWidget(self.bg_radio)
        layout.addLayout(hbox)
        # connect toggles: only act when checked
        self.fg_radio.toggled.connect(
            lambda checked: checked and self.set_brush_color(QColor("blue"))
        )
        self.bg_radio.toggled.connect(
            lambda checked: checked and self.set_brush_color(QColor("red"))
        )

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll)

        self.image_label = ImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(1, 1)
        self.image_label.setScaledContents(False)
        self.scroll.setWidget(self.image_label)
        # enable/disable Clear button based on whether painting exists
        try:
            self.image_label.paintingChanged.connect(self.clear_btn.setEnabled)
        except Exception:
            pass

        # Brush size slider (placed after image label creation)
        size_hbox = QHBoxLayout()
        size_label = QLabel("Brush size:")
        self.size_value_label = QLabel(str(self.image_label.brush_radius))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(1)
        self.size_slider.setMaximum(10)
        self.size_slider.setValue(self.image_label.brush_radius)
        self.size_slider.valueChanged.connect(self._on_size_changed)
        size_hbox.addWidget(size_label)
        size_hbox.addWidget(self.size_slider)
        size_hbox.addWidget(self.size_value_label)
        layout.addLayout(size_hbox)

    def set_brush_color(self, qcolor: QColor):
        self.image_label.set_brush_color(qcolor)

    def _on_size_changed(self, value: int):
        # update label and forward size to the image label
        self.size_value_label.setText(str(value))
        if hasattr(self, "image_label") and self.image_label is not None:
            self.image_label.set_brush_radius(value)

    def _on_clear(self):
        if hasattr(self, "image_label") and self.image_label is not None:
            self.image_label.clear_painting()

    def _on_compute(self):
        # Convert current image and painted markers into numpy arrays.
        pix = getattr(self.image_label, "_pix", None)
        orig = getattr(self.image_label, "_orig_pix", None)
        if pix is None or orig is None:
            return

        def qimage_to_array(qimg: QImage) -> np.ndarray:
            img = qimg.convertToFormat(QImage.Format_ARGB32)
            w = img.width()
            h = img.height()
            ptr = img.bits()
            # ptr is a buffer/memoryview; create a numpy array view then copy
            arr = np.frombuffer(ptr, dtype=np.uint8)
            arr = arr.reshape((h, w, 4)).copy()
            # arr is in BGRA order; convert to RGB
            rgb = arr[:, :, :3][..., ::-1].copy()
            return rgb

        img_cur = pix.toImage()
        img_orig = orig.toImage()
        arr_cur = qimage_to_array(img_cur)
        arr_orig = qimage_to_array(img_orig)

        # Produce a grayscale image array to return/print.
        if arr_cur.ndim == 3:
            # color -> convert to luminance
            arr_cur_gray = np.dot(arr_cur[..., :3], [0.299, 0.587, 0.114]).astype(
                np.uint8
            )
        else:
            arr_cur_gray = arr_cur

        if arr_orig.ndim == 3:
            arr_orig_gray = np.dot(arr_orig[..., :3], [0.299, 0.587, 0.114]).astype(
                np.uint8
            )
        else:
            arr_orig_gray = arr_orig

        # Compute marker array: 1 for foreground (blue), 2 for background (red)
        h, w = arr_cur_gray.shape[:2]
        markers = np.zeros((h, w), dtype=np.uint8)

        # detect pixels that changed due to painting
        if arr_cur.ndim == 3 and arr_orig.ndim == 3:
            changed = np.any(arr_cur != arr_orig, axis=2)
        else:
            changed = arr_cur_gray != arr_orig_gray

        # If color info is available, detect blue/red painted pixels
        if arr_cur.ndim == 3:
            blue = np.array([0, 0, 255], dtype=int)
            red = np.array([255, 0, 0], dtype=int)
            cur_int = arr_cur.astype(int)
            d_blue = np.sum((cur_int - blue) ** 2, axis=2)
            d_red = np.sum((cur_int - red) ** 2, axis=2)
            fg_mask = changed & (d_blue <= d_red)
            bg_mask = changed & (d_red < d_blue)
            markers[fg_mask] = 1
            markers[bg_mask] = 2
        else:
            # fallback: mark any changed pixel as foreground (1)
            markers[changed] = 1

        # Print grayscale image shape and marker summary
        print(
            f"image shape: {arr_cur_gray.shape}, markers unique: {np.unique(markers)}"
        )

        # Prepare image and seeds for dahu level-lines distance transform.
        # Use the original (unpainted) image so marker values do not corrupt
        # the input used for immersion/LLDT computation.
        img_for_dahu = arr_orig_gray.astype(np.uint16)

        # build fg/bg masks and immerse them
        fg_mask = (markers == 1).astype(np.uint8)
        bg_mask = (markers == 2).astype(np.uint8)

        # Compute immersion of image
        m, M = immersion(img_for_dahu)

        # Immerse the marker masks to match the notebook behavior
        _, fg_K = immersion(fg_mask)
        _, bg_K = immersion(bg_mask)

        D_fg = None
        D_bg = None
        try:
            # Extract seeds from immersed marker images (matching notebook implementation)
            if np.any(fg_K):
                seeds_fg = get_coordinates(fg_K > 0)
                _, D_fg = level_lines_distance_transform(m, M, seeds_fg)
            if np.any(bg_K):
                seeds_bg = get_coordinates(bg_K > 0)
                _, D_bg = level_lines_distance_transform(m, M, seeds_bg)
        except Exception as e:
            print("Error computing LLDT:", e)

        # Display the marker image and LLDT results in a new window.
        try:
            marker_image = get_marker_image(arr_orig_gray, markers)
        except Exception as e:
            QMessageBox.warning(
                self, "Marker error", f"Failed to build marker image:\n{e}"
            )
            return

        try:
            # Create the results window without a parent so it is independent
            dlg = ResultsWindow(marker_image, D_fg, D_bg)
            # Keep a reference so Python/GIL doesn't garbage-collect it while shown
            self._last_results_window = dlg
            # Ensure it is deleted when closed and show non-modally
            dlg.setAttribute(Qt.WA_DeleteOnClose, True)
            dlg.show()
        except Exception as e:
            QMessageBox.warning(
                self, "Display error", f"Failed to show results window:\n{e}"
            )

        return arr_cur_gray, markers

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)",
        )
        if not file_path:
            return
        qimg = QImage(file_path)
        if qimg.isNull():
            self.image_label.setText("Failed to load image.")
            return

        # If the image is color, convert it to grayscale
        if qimg.format() != QImage.Format_Grayscale8:
            try:
                qimg = qimg.convertToFormat(QImage.Format_Grayscale8)
            except Exception:
                # fallback: keep original if conversion fails
                pass

        pix = QPixmap.fromImage(qimg)
        if pix.isNull():
            self.image_label.setText("Failed to load image.")
            return
        self.image_label.setPixmap(pix)
        self.image_label.adjustSize()
        self.setWindowTitle(f"Image Viewer - {file_path}")
