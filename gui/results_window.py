import os
import numpy as np
import matplotlib.cm as cm
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QFileDialog,
    QMessageBox,
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt


class ResultsWindow(QMainWindow):
    def __init__(
        self,
        marker_image: np.ndarray,
        D_fg: np.ndarray,
        D_bg: np.ndarray,
        D_fg_geos: np.ndarray = None,
        D_bg_geos: np.ndarray = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Distance Transform Results")
        try:
            flags = self.windowFlags() | Qt.Window | Qt.WindowMinMaxButtonsHint
            self.setWindowFlags(flags)
            self.setMinimumSize(600, 400)
        except Exception:
            pass

        self.marker_image = marker_image
        self.D_fg = D_fg
        self.D_bg = D_bg
        self.D_fg_geos = D_fg_geos
        self.D_bg_geos = D_bg_geos

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        grid = QGridLayout()

        self.lbl_markers = QLabel("Markers")
        self.lbl_fg = QLabel("LLDT Foreground")
        self.lbl_bg = QLabel("LLDT Background")
        self.lbl_prob = QLabel("LLDT Probability")
        self.lbl_fg_geos = QLabel("Geodesic Foreground")
        self.lbl_bg_geos = QLabel("Geodesic Background")
        self.lbl_prob_geos = QLabel("Geodesic Probability")

        self.img_markers = QLabel()
        self.img_markers.setAlignment(Qt.AlignCenter)
        self.img_fg = QLabel()
        self.img_fg.setAlignment(Qt.AlignCenter)
        self.img_bg = QLabel()
        self.img_bg.setAlignment(Qt.AlignCenter)
        self.img_prob = QLabel()
        self.img_prob.setAlignment(Qt.AlignCenter)
        self.img_fg_geos = QLabel()
        self.img_fg_geos.setAlignment(Qt.AlignCenter)
        self.img_bg_geos = QLabel()
        self.img_bg_geos.setAlignment(Qt.AlignCenter)
        self.img_prob_geos = QLabel()
        self.img_prob_geos.setAlignment(Qt.AlignCenter)

        # First row: LLDT results
        grid.addWidget(self.lbl_markers, 0, 0)
        grid.addWidget(self.lbl_fg, 0, 1)
        grid.addWidget(self.lbl_bg, 0, 2)
        grid.addWidget(self.lbl_prob, 0, 3)
        grid.addWidget(self.img_markers, 1, 0)
        grid.addWidget(self.img_fg, 1, 1)
        grid.addWidget(self.img_bg, 1, 2)
        grid.addWidget(self.img_prob, 1, 3)

        # Second row: Geodesic results
        grid.addWidget(self.lbl_fg_geos, 2, 1)
        grid.addWidget(self.lbl_bg_geos, 2, 2)
        grid.addWidget(self.lbl_prob_geos, 2, 3)
        grid.addWidget(self.img_fg_geos, 3, 1)
        grid.addWidget(self.img_bg_geos, 3, 2)
        grid.addWidget(self.img_prob_geos, 3, 3)

        layout.addLayout(grid)

        btn_h = QHBoxLayout()
        self.save_btn = QPushButton("Save Images")
        self.save_btn.clicked.connect(self._on_save)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btn_h.addWidget(self.save_btn)
        btn_h.addWidget(self.close_btn)
        layout.addLayout(btn_h)

        self._update_display()

    def _normalize_to_uint8(self, arr: np.ndarray) -> np.ndarray:
        if arr is None:
            return None
        a = arr.astype(np.float64)
        mn = a.min()
        mx = a.max()
        if mx == mn:
            return np.zeros(a.shape, dtype=np.uint8)
        norm = ((a - mn) / (mx - mn) * 255.0).astype(np.uint8)
        return norm

    def _array_to_qimage(self, arr: np.ndarray) -> QImage:
        if arr is None:
            return QImage()
        if arr.ndim == 2:
            h, w = arr.shape
            arr2 = arr.copy()
            return QImage(arr2.data, w, h, w, QImage.Format_Grayscale8).copy()
        elif arr.ndim == 3 and arr.shape[2] == 3:
            h, w, _ = arr.shape
            arr2 = arr.copy()
            return QImage(arr2.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        else:
            return QImage()

    def _update_display(self):
        # build original pixmaps from arrays (keep originals for scaling)
        q_markers = self._array_to_qimage(self.marker_image)
        self._pix_markers = (
            QPixmap.fromImage(q_markers) if not q_markers.isNull() else None
        )

        # use inferno colormap for LLDT displays and store original pixmaps
        cmap = cm.get_cmap("inferno")
        cmap_r = cm.get_cmap("inferno_r")
        self._pix_fg = None
        self._pix_bg = None
        self._pix_prob = None
        if self.D_fg is not None:
            fg_norm = self._normalize_to_uint8(self.D_fg)
            fg_rgb = (cmap(fg_norm / 255.0)[..., :3] * 255).astype(np.uint8)
            qfg = self._array_to_qimage(fg_rgb)
            self._pix_fg = QPixmap.fromImage(qfg)
        if self.D_bg is not None:
            bg_norm = self._normalize_to_uint8(self.D_bg)
            bg_rgb = (cmap(bg_norm / 255.0)[..., :3] * 255).astype(np.uint8)
            qbg = self._array_to_qimage(bg_rgb)
            self._pix_bg = QPixmap.fromImage(qbg)

        # Compute probability: D_fg / (D_fg + D_bg)
        if self.D_fg is not None and self.D_bg is not None:
            # Avoid division by zero
            denom = self.D_fg.astype(np.float64) + self.D_bg.astype(np.float64)
            prob = np.zeros_like(self.D_fg, dtype=np.float64)
            mask = denom > 0
            prob[mask] = self.D_fg[mask].astype(np.float64) / denom[mask]
            # Normalize to [0, 255]
            prob_uint8 = (prob * 255).astype(np.uint8)
            prob_rgb = (cmap_r(prob)[..., :3] * 255).astype(np.uint8)
            qprob = self._array_to_qimage(prob_rgb)
            self._pix_prob = QPixmap.fromImage(qprob)

        # Add geodesic distance transforms
        self._pix_fg_geos = None
        self._pix_bg_geos = None
        self._pix_prob_geos = None
        if self.D_fg_geos is not None:
            fg_geos_norm = self._normalize_to_uint8(self.D_fg_geos)
            fg_geos_rgb = (cmap(fg_geos_norm / 255.0)[..., :3] * 255).astype(np.uint8)
            qfg_geos = self._array_to_qimage(fg_geos_rgb)
            self._pix_fg_geos = QPixmap.fromImage(qfg_geos)
        if self.D_bg_geos is not None:
            bg_geos_norm = self._normalize_to_uint8(self.D_bg_geos)
            bg_geos_rgb = (cmap(bg_geos_norm / 255.0)[..., :3] * 255).astype(np.uint8)
            qbg_geos = self._array_to_qimage(bg_geos_rgb)
            self._pix_bg_geos = QPixmap.fromImage(qbg_geos)

        # Compute geodesic probability: D_fg_geos / (D_fg_geos + D_bg_geos)
        if self.D_fg_geos is not None and self.D_bg_geos is not None:
            denom_geos = self.D_fg_geos.astype(np.float64) + self.D_bg_geos.astype(
                np.float64
            )
            prob_geos = np.zeros_like(self.D_fg_geos, dtype=np.float64)
            mask_geos = denom_geos > 0
            prob_geos[mask_geos] = (
                self.D_fg_geos[mask_geos].astype(np.float64) / denom_geos[mask_geos]
            )
            prob_geos_rgb = (cmap_r(prob_geos)[..., :3] * 255).astype(np.uint8)
            qprob_geos = self._array_to_qimage(prob_geos_rgb)
            self._pix_prob_geos = QPixmap.fromImage(qprob_geos)

        # scale to the current label sizes
        self._rescale_pixmaps()

    def _on_save(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select empty directory to save images"
        )
        if not d:
            return
        # ensure directory is empty
        if os.listdir(d):
            QMessageBox.warning(
                self, "Directory not empty", "Please select an empty directory."
            )
            return
        # save marker image (RGB)
        try:
            q_markers = self._array_to_qimage(self.marker_image)
            q_markers.save(os.path.join(d, "markers.png"))
        except Exception as e:
            QMessageBox.warning(self, "Save error", f"Failed to save markers: {e}")
            return
        # save fg and bg if present
        try:
            cmap = cm.get_cmap("inferno")
            if self.D_fg is not None:
                fg_norm = self._normalize_to_uint8(self.D_fg)
                fg_rgb = (cmap(fg_norm / 255.0)[..., :3] * 255).astype(np.uint8)
                qfg = self._array_to_qimage(fg_rgb)
                qfg.save(os.path.join(d, "fg.png"))
            if self.D_bg is not None:
                bg_norm = self._normalize_to_uint8(self.D_bg)
                bg_rgb = (cmap(bg_norm / 255.0)[..., :3] * 255).astype(np.uint8)
                qbg = self._array_to_qimage(bg_rgb)
                qbg.save(os.path.join(d, "bg.png"))

            # Save probability image
            if self.D_fg is not None and self.D_bg is not None:
                cmap_r = cm.get_cmap("inferno_r")
                denom = self.D_fg.astype(np.float64) + self.D_bg.astype(np.float64)
                prob = np.zeros_like(self.D_fg, dtype=np.float64)
                mask = denom > 0
                prob[mask] = self.D_fg[mask].astype(np.float64) / denom[mask]
                prob_rgb = (cmap_r(prob)[..., :3] * 255).astype(np.uint8)
                qprob = self._array_to_qimage(prob_rgb)
                qprob.save(os.path.join(d, "probability.png"))

            # Save geodesic distance transforms
            if self.D_fg_geos is not None:
                fg_geos_norm = self._normalize_to_uint8(self.D_fg_geos)
                fg_geos_rgb = (cmap(fg_geos_norm / 255.0)[..., :3] * 255).astype(
                    np.uint8
                )
                qfg_geos = self._array_to_qimage(fg_geos_rgb)
                qfg_geos.save(os.path.join(d, "fg_geos.png"))
            if self.D_bg_geos is not None:
                bg_geos_norm = self._normalize_to_uint8(self.D_bg_geos)
                bg_geos_rgb = (cmap(bg_geos_norm / 255.0)[..., :3] * 255).astype(
                    np.uint8
                )
                qbg_geos = self._array_to_qimage(bg_geos_rgb)
                qbg_geos.save(os.path.join(d, "bg_geos.png"))

            # Save geodesic probability image
            if self.D_fg_geos is not None and self.D_bg_geos is not None:
                cmap_r = cm.get_cmap("inferno_r")
                denom_geos = self.D_fg_geos.astype(np.float64) + self.D_bg_geos.astype(
                    np.float64
                )
                prob_geos = np.zeros_like(self.D_fg_geos, dtype=np.float64)
                mask_geos = denom_geos > 0
                prob_geos[mask_geos] = (
                    self.D_fg_geos[mask_geos].astype(np.float64) / denom_geos[mask_geos]
                )
                prob_geos_rgb = (cmap_r(prob_geos)[..., :3] * 255).astype(np.uint8)
                qprob_geos = self._array_to_qimage(prob_geos_rgb)
                qprob_geos.save(os.path.join(d, "probability_geos.png"))
        except Exception as e:
            QMessageBox.warning(
                self, "Save error", f"Failed to save distance transforms: {e}"
            )
            return

        QMessageBox.information(self, "Saved", f"Images saved to {d}")

    def _rescale_pixmaps(self):
        # Scale stored original pixmaps to the label sizes while keeping aspect ratio
        try:
            if getattr(self, "_pix_markers", None) is not None:
                target = self.img_markers.size()
                if target.width() > 0 and target.height() > 0:
                    self.img_markers.setPixmap(
                        self._pix_markers.scaled(
                            target, Qt.KeepAspectRatio, Qt.SmoothTransformation
                        )
                    )
            if getattr(self, "_pix_fg", None) is not None:
                target = self.img_fg.size()
                if target.width() > 0 and target.height() > 0:
                    self.img_fg.setPixmap(
                        self._pix_fg.scaled(
                            target, Qt.KeepAspectRatio, Qt.SmoothTransformation
                        )
                    )
            if getattr(self, "_pix_bg", None) is not None:
                target = self.img_bg.size()
                if target.width() > 0 and target.height() > 0:
                    self.img_bg.setPixmap(
                        self._pix_bg.scaled(
                            target, Qt.KeepAspectRatio, Qt.SmoothTransformation
                        )
                    )
            if getattr(self, "_pix_prob", None) is not None:
                target = self.img_prob.size()
                if target.width() > 0 and target.height() > 0:
                    self.img_prob.setPixmap(
                        self._pix_prob.scaled(
                            target, Qt.KeepAspectRatio, Qt.SmoothTransformation
                        )
                    )
            if getattr(self, "_pix_fg_geos", None) is not None:
                target = self.img_fg_geos.size()
                if target.width() > 0 and target.height() > 0:
                    self.img_fg_geos.setPixmap(
                        self._pix_fg_geos.scaled(
                            target, Qt.KeepAspectRatio, Qt.SmoothTransformation
                        )
                    )
            if getattr(self, "_pix_bg_geos", None) is not None:
                target = self.img_bg_geos.size()
                if target.width() > 0 and target.height() > 0:
                    self.img_bg_geos.setPixmap(
                        self._pix_bg_geos.scaled(
                            target, Qt.KeepAspectRatio, Qt.SmoothTransformation
                        )
                    )
            if getattr(self, "_pix_prob_geos", None) is not None:
                target = self.img_prob_geos.size()
                if target.width() > 0 and target.height() > 0:
                    self.img_prob_geos.setPixmap(
                        self._pix_prob_geos.scaled(
                            target, Qt.KeepAspectRatio, Qt.SmoothTransformation
                        )
                    )
        except Exception:
            pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Rescale images when window is resized or maximized
        self._rescale_pixmaps()
