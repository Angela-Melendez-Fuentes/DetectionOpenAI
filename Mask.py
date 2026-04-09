import sys
import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap

# =========================================================
# 1. FUNCIONES DE PREPROCESAMIENTO Y ESCANEO
# =========================================================
def redimensionar(img, height=800):
    aspect_ratio = img.shape[1] / img.shape[0]
    width_target = int(height * aspect_ratio)
    return cv2.resize(img, (width_target, height))

def extraer_documento(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)
    
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
            d = np.diff(pts, axis=1); rect[1] = pts[np.argmin(d)]; rect[3] = pts[np.argmax(d)]
            
            # Forzamos una resolución final (600x800) para homogeneizar los filtros
            dst = np.array([[0,0],[599,0],[599,799],[0,799]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            return cv2.warpPerspective(img, M, (600, 800)), True
            
    return img, False

# =========================================================
# 2. INTERFAZ GRÁFICA DEL CALIBRADOR
# =========================================================
class VisorImagen(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()  
        self.setFrameShape(QtWidgets.QFrame.Shape.Box) 
        self.setLineWidth(1)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 500)

class CalibradorMascaras(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Laboratorio de Calibración OpenCV (Hojas Blancas + Cuaderno)")
        self.resize(1300, 800) 
        self.OpenCV_image = None  
        self.doc_recortado = None
        self.crear_widgets()
        self.configurar_layout()

    def crear_widgets(self):
        self.botonAbrir = QtWidgets.QPushButton("Abrir Imagen")
        self.botonAbrir.setStyleSheet("background-color: #D5FFCC; color: #4D941E; font-weight: bold; border-radius: 5px;")
        self.botonAbrir.setMinimumHeight(50)
        self.botonAbrir.clicked.connect(self.handleOpen)

        self.crear_panel_controles()

        self.viewer_original = VisorImagen()  
        self.viewer_mascara1 = VisorImagen()
        self.viewer_mascara2 = VisorImagen()
        self.viewer_resultado = VisorImagen()
        self.viewer_cajas = VisorImagen()

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.viewer_mascara1, "1. Texto Sucio (Threshold)")
        self.tabs.addTab(self.viewer_mascara2, "2. Líneas Detectadas")
        self.tabs.addTab(self.viewer_resultado, "3. Máscara Limpia")
        self.tabs.addTab(self.viewer_cajas, "4. Cajas de Párrafos")

    def crear_panel_controles(self):
        self.panel_controles = QtWidgets.QGroupBox("Controles en Tiempo Real")
        layout = QtWidgets.QVBoxLayout(self.panel_controles)

        # --- CHECKBOX DE TIPO DE PAPEL ---
        self.check_cuadricula = QtWidgets.QCheckBox("El papel tiene cuadrícula/rayas (Activar resta de líneas)")
        self.check_cuadricula.setChecked(True) # Activado por defecto
        self.check_cuadricula.setStyleSheet("font-weight: bold; color: #b30000; padding: 5px;")

        # 1. Block Size (Adaptive Threshold) - Default: 11
        self.lbl_bs = QtWidgets.QLabel("Sensibilidad de Tinta (Block Size): 11")
        self.slider_bs = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider_bs.setRange(3, 199)
        self.slider_bs.setSingleStep(2)
        self.slider_bs.setValue(11)

        # 2. Sensibilidad de Borrado de Líneas - Default: 14
        self.lbl_lineas = QtWidgets.QLabel("Filtro de Cuadrícula (Longitud): 14")
        self.slider_lineas = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider_lineas.setRange(2, 150)
        self.slider_lineas.setValue(14)

        # 3. Ancho de Párrafo - Default: 13
        self.lbl_pw = QtWidgets.QLabel("Agrupar Horizontalmente (Ancho): 13")
        self.slider_pw = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider_pw.setRange(2, 200)
        self.slider_pw.setValue(13)

        # 4. Alto de Párrafo - Default: 2
        self.lbl_ph = QtWidgets.QLabel("Agrupar Verticalmente (Alto): 2")
        self.slider_ph = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider_ph.setRange(2, 150)
        self.slider_ph.setValue(2)

        layout.addWidget(self.check_cuadricula)
        layout.addWidget(self.lbl_bs)
        layout.addWidget(self.slider_bs)
        layout.addWidget(self.lbl_lineas)
        layout.addWidget(self.slider_lineas)
        layout.addWidget(self.lbl_pw)
        layout.addWidget(self.slider_pw)
        layout.addWidget(self.lbl_ph)
        layout.addWidget(self.slider_ph)

        # Conectar TODO a la actualización
        self.check_cuadricula.stateChanged.connect(self.actualizar_procesamiento)
        self.slider_bs.valueChanged.connect(self.actualizar_procesamiento)
        self.slider_lineas.valueChanged.connect(self.actualizar_procesamiento)
        self.slider_pw.valueChanged.connect(self.actualizar_procesamiento)
        self.slider_ph.valueChanged.connect(self.actualizar_procesamiento)

    def configurar_layout(self):
        layout_principal = QtWidgets.QVBoxLayout(self)
        
        layout_top = QtWidgets.QHBoxLayout()
        layout_top.addWidget(self.botonAbrir)
        layout_top.addWidget(self.panel_controles)
        layout_principal.addLayout(layout_top)
        
        layout_cuerpo = QtWidgets.QHBoxLayout()
        layout_izq = QtWidgets.QVBoxLayout()
        self.lbl_izq = QtWidgets.QLabel("DOCUMENTO ESCANEADO")
        self.lbl_izq.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_izq.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Weight.Bold))
        layout_izq.addWidget(self.lbl_izq)
        layout_izq.addWidget(self.viewer_original)
        
        layout_cuerpo.addLayout(layout_izq, 1)        
        layout_cuerpo.addWidget(self.tabs, 2) 
        layout_principal.addLayout(layout_cuerpo)

    def handleOpen(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Seleccionar Imagen", ".", "Imágenes (*.jpg *.png *.jpeg)")
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.OpenCV_image = redimensionar(img)
                doc_transformado, _ = extraer_documento(self.OpenCV_image)
                self.doc_recortado = doc_transformado
                self.ActualizarPixMap(self.viewer_original, self.doc_recortado)
                self.actualizar_procesamiento()

    def actualizar_procesamiento(self):
        if self.doc_recortado is None: return

        bs = self.slider_bs.value()
        if bs % 2 == 0: bs += 1
        
        len_lineas = self.slider_lineas.value()
        pw = self.slider_pw.value()
        ph = self.slider_ph.value()

        self.lbl_bs.setText(f"Sensibilidad de Tinta (Block Size): {bs} (Impar)")
        self.lbl_lineas.setText(f"Filtro de Cuadrícula (Longitud): {len_lineas}")
        self.lbl_pw.setText(f"Agrupar Horizontalmente (Ancho): {pw}")
        self.lbl_ph.setText(f"Agrupar Verticalmente (Alto): {ph}")

        # Si no hay cuadrícula, deshabilitar visualmente el slider de líneas
        self.slider_lineas.setEnabled(self.check_cuadricula.isChecked())

        # --- 1. MÁSCARA 1 (Texto Sucio) ---
        gray = cv2.cvtColor(self.doc_recortado, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        gray = cv2.medianBlur(gray, 3) 
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bs, 10)
        
        # --- 2. LÓGICA CONDICIONAL DE MÁSCARA 2 ---
        if self.check_cuadricula.isChecked():
            # Hoja de Cuaderno: Buscar líneas y restarlas
            hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (len_lineas, 1))
            ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, len_lineas))
            lineas_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, hor_kernel, iterations=1)
            lineas_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, ver_kernel, iterations=1)
            mascara_lineas = cv2.add(lineas_h, lineas_v)
            mascara_limpia = cv2.subtract(thresh, mascara_lineas)
        else:
            # Hoja Blanca: No restar nada, usar el threshold original
            mascara_lineas = np.zeros_like(thresh) # Imagen en negro para el visor
            mascara_limpia = thresh.copy()

        # --- 3. DETECCIÓN DE PÁRRAFOS ---
        k_parrafo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pw, ph)) 
        mask_p = cv2.dilate(mascara_limpia, k_parrafo, iterations=1)
        cnts_p, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        img_cajas = self.doc_recortado.copy()
        for c in cnts_p:
            x, y, w, h = cv2.boundingRect(c)
            if w > 15 and h > 15: 
                cv2.rectangle(img_cajas, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Mostrar en visores
        self.ActualizarPixMap(self.viewer_mascara1, cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))
        self.ActualizarPixMap(self.viewer_mascara2, cv2.cvtColor(mascara_lineas, cv2.COLOR_GRAY2BGR))
        self.ActualizarPixMap(self.viewer_resultado, cv2.cvtColor(mascara_limpia, cv2.COLOR_GRAY2BGR))
        self.ActualizarPixMap(self.viewer_cajas, img_cajas)

    def ActualizarPixMap(self, label_target, image):
        if image is None: return
        h, w, _ = image.shape
        qImg = QImage(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).data, w, h, 3*w, QImage.Format.Format_RGB888)
        label_target.setPixmap(QPixmap.fromImage(qImg).scaled(label_target.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = CalibradorMascaras()
    window.show()
    sys.exit(app.exec())