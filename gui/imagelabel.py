from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap, QPainter, QColor
from PySide6.QtCore import Qt, Signal


class ImageLabel(QLabel):
    paintingChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pix = None
        self._orig_pix = None
        self._has_painting = False
        self.drawing = False
        self.brush_color = QColor("blue")
        self.brush_radius = 8

    def set_brush_radius(self, r: int):
        if r < 1:
            r = 1
        self.brush_radius = int(r)

    def set_brush_color(self, color: QColor):
        self.brush_color = color

    def setPixmap(self, pixmap: QPixmap):
        # Keep an editable copy of the pixmap
        if pixmap is None:
            self._pix = None
            self._orig_pix = None
            super().setPixmap(pixmap)
            return
        # keep original and editable copy
        self._orig_pix = pixmap.copy()
        self._pix = self._orig_pix.copy()
        # reset painting state for new image
        self._has_painting = False
        try:
            self.paintingChanged.emit(False)
        except Exception:
            pass
        super().setPixmap(self._pix)

    def clear_painting(self):
        """Restore the image to the original loaded pixmap (clears painting)."""
        if self._orig_pix is None:
            return
        self._pix = self._orig_pix.copy()
        super().setPixmap(self._pix)
        if getattr(self, "_has_painting", False):
            self._has_painting = False
            try:
                self.paintingChanged.emit(False)
            except Exception:
                pass

    def _map_to_pix(self, pos):
        if self._pix is None:
            return None
        pix_w = self._pix.width()
        pix_h = self._pix.height()
        lbl_w = self.width()
        lbl_h = self.height()
        x_off = max(0, (lbl_w - pix_w) // 2)
        y_off = max(0, (lbl_h - pix_h) // 2)
        x = pos.x() - x_off
        y = pos.y() - y_off
        if 0 <= x < pix_w and 0 <= y < pix_h:
            return x, y
        return None

    def _paint_at(self, x, y):
        if self._pix is None:
            return
        painter = QPainter(self._pix)
        painter.setBrush(self.brush_color)
        painter.setPen(Qt.NoPen)
        r = self.brush_radius
        painter.drawEllipse(x - r, y - r, r * 2, r * 2)
        painter.end()
        super().setPixmap(self._pix)
        if not getattr(self, "_has_painting", False):
            self._has_painting = True
            try:
                self.paintingChanged.emit(True)
            except Exception:
                pass

    def mousePressEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            mapped = self._map_to_pix(event.pos())
            if mapped:
                self.drawing = True
                x, y = mapped
                self._paint_at(x, y)

    def mouseMoveEvent(self, event):
        if not self.drawing:
            return
        mapped = self._map_to_pix(event.pos())
        if mapped:
            x, y = mapped
            self._paint_at(x, y)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
