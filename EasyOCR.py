import sys
import cv2
import numpy as np
import easyocr
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap

# =========================================================
# 1. FUNCIONES DE PREPROCESAMIENTO
# =========================================================
def redimensionar(img, height=800):
    aspect_ratio = img.shape[1] / img.shape[0]
    width_target = int(height * aspect_ratio)
    img_resized = cv2.resize(img, (width_target, height))
    return img_resized

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
            
            dst = np.array([[0,0],[599,0],[599,799],[0,799]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            return cv2.warpPerspective(img, M, (600, 800)), True
            
    return img, False

def limpiar_fondo_y_cuadricula(img_color):
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    lineas_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, hor_kernel, iterations=2)
    lineas_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, ver_kernel, iterations=2)
    
    cuadricula = lineas_h + lineas_v
    limpia = cv2.subtract(thresh, cuadricula)
    
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    limpia = cv2.morphologyEx(limpia, cv2.MORPH_OPEN, kernel_clean)
    
    limpia_inv = cv2.bitwise_not(limpia)
    limpia_color = cv2.cvtColor(limpia_inv, cv2.COLOR_GRAY2BGR)
    return limpia_color

# =========================================================
# 2. HILO PARA EASYOCR (LÍNEA POR LÍNEA)
# =========================================================
class WorkerEasyOCR(QtCore.QThread):
    linea_procesada = QtCore.pyqtSignal(int, np.ndarray, str)
    proceso_terminado = QtCore.pyqtSignal()
    error_detectado = QtCore.pyqtSignal(str)

    def __init__(self, lista_recortes_lineas):
        super().__init__()
        self.lista_recortes_lineas = lista_recortes_lineas

    def run(self):
        try:
            # Inicializamos el lector de EasyOCR (en español)
            lector_ocr = easyocr.Reader(['es'])
            
            for index, img_linea in enumerate(self.lista_recortes_lineas):
                # Extraer texto de la línea recortada usando detail=0
                resultados = lector_ocr.readtext(img_linea, detail=0)
                texto_linea = " ".join(resultados)
                
                self.linea_procesada.emit(index, img_linea, texto_linea)
            
            self.proceso_terminado.emit()
        except Exception as e:
            self.error_detectado.emit(f"Error con EasyOCR:\n{str(e)}")

# =========================================================
# 3. INTERFAZ GRÁFICA Y CONTROLADOR
# =========================================================
class VisorImagen(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()  
        self.setFrameShape(QtWidgets.QFrame.Shape.Box) 
        self.setLineWidth(2)
        self.setScaledContents(False) 
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumSize(350, 400)

class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector de Escritura a Mano + EasyOCR")
        self.resize(1500, 900) 
        
        self.OpenCV_image = None  
        self.procesedImage = None 
        self.img_lineas = None
        self.img_caracteres = None
        
        self._path = None
        self.palabra_count = 0    
        self.palabra_count_ocr = 0 
        self.caracteres_count_ocr = 0 
        self.worker = None 
        
        self.crear_widgets()
        self.configurar_layout()
        self.conectar_senales()

    def crear_widgets(self):
        self.botonAbrir = QtWidgets.QPushButton("Abrir Imagen")
        self.botonAbrir.setStyleSheet("background-color: #D5FFCC; color: #4D941E; font-weight: bold; border: 1px solid #4D941E; border-radius: 5px;")

        self.botonProcesarImagenEntrada = QtWidgets.QPushButton("Analizar y Transcribir por Líneas (EasyOCR)")
        self.botonProcesarImagenEntrada.setStyleSheet("background-color: #CCEDFF; color: #1E5C94; font-weight: bold; border: 1px solid #1E5C94; border-radius: 5px;")

        self.botonLimpiar = QtWidgets.QPushButton("Limpiar")
        self.botonLimpiar.setStyleSheet("background-color: #ffcccc; color: #cc0000; font-weight: bold; border: 1px solid #cc0000; border-radius: 5px;")
        
        for btn in [self.botonAbrir, self.botonProcesarImagenEntrada, self.botonLimpiar]:
            btn.setMinimumHeight(50) 
            btn.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Weight.Bold))
            btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

        self.botonProcesarImagenEntrada.setEnabled(False)
        self.botonLimpiar.setEnabled(False)
        
        self.viewer = VisorImagen()   
        self.viewer2 = VisorImagen()  
        self.viewer3 = VisorImagen()  
        
        self.visorTexto = QtWidgets.QTextEdit()
        self.visorTexto.setReadOnly(True)
        self.visorTexto.setFont(QtGui.QFont("Arial", 14))
        self.visorTexto.setStyleSheet("background-color: #f9f9f9; border: 2px solid #cccccc; padding: 10px;")
        
        self.label_contador_cv = QtWidgets.QLabel("Caracteres detectados: 0")
        self.label_contador_cv.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_contador_cv.setStyleSheet("background-color: #e6f2ff; border: 1px solid #99ccff; color: #003366;")
        self.label_contador_cv.setMinimumHeight(40)
        self.label_contador_cv.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Weight.Bold))

        self.label_contador_ocr = QtWidgets.QLabel("Palabras transcritas: 0 | Caracteres: 0")
        self.label_contador_ocr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_contador_ocr.setStyleSheet("background-color: #e6f2ff; border: 1px solid #99ccff; color: #003366;")
        self.label_contador_ocr.setMinimumHeight(40)
        self.label_contador_ocr.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Weight.Bold))

        self.tabs_derecha = QtWidgets.QTabWidget()
        self.tabs_derecha.setStyleSheet("QTabBar::tab { height: 40px; padding: 0 20px; font-weight: bold; font-size: 11pt; }")

        # Pestaña 1: Línea por Línea
        self.tab_lineas_ocr = QtWidgets.QWidget()
        layout_lineas_ocr = QtWidgets.QVBoxLayout(self.tab_lineas_ocr)
        
        self.scroll_ocr = QtWidgets.QScrollArea()
        self.scroll_ocr.setWidgetResizable(True)
        self.scroll_widget = QtWidgets.QWidget()
        self.scroll_vbox = QtWidgets.QVBoxLayout(self.scroll_widget)
        self.scroll_vbox.setAlignment(Qt.AlignmentFlag.AlignTop) 
        self.scroll_ocr.setWidget(self.scroll_widget)
        
        layout_lineas_ocr.addWidget(self.scroll_ocr)
        layout_lineas_ocr.addWidget(self.label_contador_ocr)

        # Pestaña 2: Texto Completo
        self.tab_transcripcion = QtWidgets.QWidget()
        layout_trans = QtWidgets.QVBoxLayout(self.tab_transcripcion)
        layout_trans.addWidget(self.visorTexto)
        
        # Pestaña 3: Líneas de OpenCV
        self.tab_lineas = QtWidgets.QWidget()
        layout_lineas = QtWidgets.QVBoxLayout(self.tab_lineas)
        layout_lineas.addWidget(self.viewer2)
        
        # Pestaña 4: Caracteres de OpenCV
        self.tab_caracteres = QtWidgets.QWidget()
        layout_caracteres = QtWidgets.QVBoxLayout(self.tab_caracteres)
        layout_caracteres.addWidget(self.viewer3)
        layout_caracteres.addWidget(self.label_contador_cv)
        
        self.tabs_derecha.addTab(self.tab_lineas_ocr, "📄 EasyOCR: Línea por Línea")
        self.tabs_derecha.addTab(self.tab_transcripcion, "📝 EasyOCR: Texto Completo")
        self.tabs_derecha.addTab(self.tab_lineas, "⬛ OpenCV: Líneas")
        self.tabs_derecha.addTab(self.tab_caracteres, "🟩 OpenCV: Caracteres")

    def configurar_layout(self):
        layout_principal = QtWidgets.QVBoxLayout(self)
        layout_principal.setContentsMargins(20, 20, 20, 20) 
        layout_principal.setSpacing(15)
        
        layout_botones = QtWidgets.QHBoxLayout()
        layout_botones.addWidget(self.botonAbrir)
        layout_botones.addWidget(self.botonProcesarImagenEntrada)
        layout_botones.addWidget(self.botonLimpiar)
        layout_principal.addLayout(layout_botones)

        layout_cuerpo = QtWidgets.QHBoxLayout()
        
        layout_izq = QtWidgets.QVBoxLayout()
        lbl_izq = QtWidgets.QLabel("IMAGEN ORIGINAL")
        lbl_izq.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_izq.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Weight.Bold))
        lbl_izq.setStyleSheet("color: #333; background-color: #e6e6e6; padding: 10px; border-radius: 3px;")
        
        layout_izq.addWidget(lbl_izq)
        layout_izq.addWidget(self.viewer)
        
        layout_cuerpo.addLayout(layout_izq, 1)        
        layout_cuerpo.addWidget(self.tabs_derecha, 1) 

        layout_principal.addLayout(layout_cuerpo)

    def conectar_senales(self):
        self.botonAbrir.clicked.connect(self.handleOpen)
        self.botonProcesarImagenEntrada.clicked.connect(self.procesar_todo)
        self.botonLimpiar.clicked.connect(self.handleLimpiar)

    def actualizar_contadores(self):
        self.label_contador_cv.setText(f"Caracteres detectados: {self.palabra_count}")
        self.label_contador_ocr.setText(f"Palabras transcritas: {self.palabra_count_ocr} | Caracteres: {self.caracteres_count_ocr}")

    def limpiar_lista_lineas(self):
        for i in reversed(range(self.scroll_vbox.count())): 
            widget_to_remove = self.scroll_vbox.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)
                widget_to_remove.deleteLater()

    def handleOpen(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Seleccionar Imagen", ".", "Imágenes (*.jpg *.png *.jpeg *.bmp)")
        if path:
            self._path = path
            img = cv2.imread(path)
            if img is not None:
                self.OpenCV_image = img
                self.ActualizarPixMap(self.viewer, self.OpenCV_image)
                self.botonProcesarImagenEntrada.setEnabled(True)
                self.botonLimpiar.setEnabled(True)
                self.viewer2.clear()
                self.viewer3.clear()
                self.visorTexto.clear()
                self.limpiar_lista_lineas()
                
                self.palabra_count = 0
                self.palabra_count_ocr = 0
                self.caracteres_count_ocr = 0
                self.actualizar_contadores()
                
                self.tabs_derecha.setCurrentIndex(0)

    def procesar_todo(self):
        if self.OpenCV_image is None: return
        self.limpiar_lista_lineas()
        self.visorTexto.clear()
        self.botonProcesarImagenEntrada.setEnabled(False) 
        
        img_res = redimensionar(self.OpenCV_image)
        doc, detectado = extraer_documento(img_res)
        self.procesedImage = limpiar_fondo_y_cuadricula(doc) 
        
        self.img_lineas, self.img_caracteres, recortes = self.analizar_lineas_y_caracteres(self.procesedImage)
        
        self.ActualizarPixMap(self.viewer2, self.img_lineas)
        self.ActualizarPixMap(self.viewer3, self.img_caracteres)
        
        self.worker = WorkerEasyOCR(recortes)
        self.worker.linea_procesada.connect(self.mostrar_resultado_linea)
        self.worker.proceso_terminado.connect(self.finalizar_proceso)
        self.worker.error_detectado.connect(self.mostrar_error)
        self.worker.start()

    def analizar_lineas_y_caracteres(self, img_limpia_color):
        gray = cv2.cvtColor(img_limpia_color, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        img_lineas_visual = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
        img_caracteres_visual = img_limpia_color.copy()

        kernel_lineas = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 5))
        mask_lineas = cv2.dilate(bin_img, kernel_lineas, iterations=1)
        
        contornos_lineas, _ = cv2.findContours(mask_lineas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes_lineas = [cv2.boundingRect(c) for c in contornos_lineas]
        bboxes_lineas = [b for b in bboxes_lineas if b[2] > 30 and b[3] > 15]
        bboxes_lineas = sorted(bboxes_lineas, key=lambda b: b[1]) 
        
        self.palabra_count = 0
        lista_recortes = []

        for x, y, w, h in bboxes_lineas:
            cv2.rectangle(img_lineas_visual, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            recorte_color = img_limpia_color[y:y+h, x:x+w]
            lista_recortes.append(recorte_color)

            roi_linea = bin_img[y:y+h, x:x+w]
            kernel_char = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            mask_char = cv2.dilate(roi_linea, kernel_char, iterations=1)
            contornos_char, _ = cv2.findContours(mask_char, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for char_cnt in contornos_char:
                cx, cy, cw, ch = cv2.boundingRect(char_cnt)
                if cw > 5 and ch > 10:
                    self.palabra_count += 1
                    cv2.rectangle(img_caracteres_visual, (x + cx, y + cy), (x + cx + cw, y + cy + ch), (0, 200, 0), 2)

        self.actualizar_contadores()
        return img_lineas_visual, img_caracteres_visual, lista_recortes

    def mostrar_resultado_linea(self, index, img_linea, texto):
        group_box = QtWidgets.QGroupBox(f"LÍNEA {index + 1}:")
        group_box.setStyleSheet("QGroupBox { font-weight: bold; color: #1E5C94; margin-top: 10px; }")
        layout_h = QtWidgets.QHBoxLayout(group_box)
        
        lbl_img = QtWidgets.QLabel()
        lbl_img.setStyleSheet("border: 1px solid #aaa; background-color: white;")
        height, width, channel = img_linea.shape
        bytesPerLine = 3 * width
        rgb_image = cv2.cvtColor(img_linea, cv2.COLOR_BGR2RGB)
        qImg = QImage(rgb_image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        lbl_img.setPixmap(pixmap.scaledToHeight(50, Qt.TransformationMode.SmoothTransformation))
        
        lbl_texto = QtWidgets.QTextEdit()
        lbl_texto.setReadOnly(True)
        lbl_texto.setText(f"> Texto EasyOCR:\n{texto}")
        lbl_texto.setMaximumHeight(60)
        lbl_texto.setStyleSheet("background-color: #f0f8ff; font-size: 13px; color: #333; border: 1px solid #ccc;")
        
        layout_h.addWidget(lbl_img)
        layout_h.addWidget(lbl_texto)
        layout_h.setStretch(0, 1)
        layout_h.setStretch(1, 3)
        
        self.scroll_vbox.addWidget(group_box)
        
        texto_actual = self.visorTexto.toPlainText()
        self.visorTexto.setText(texto_actual + texto + "\n")
        
        palabras_ocr = texto.split()
        self.palabra_count_ocr += len(palabras_ocr)
        
        texto_limpio = texto.replace(" ", "").replace("\n", "")
        self.caracteres_count_ocr += len(texto_limpio)
        
        self.actualizar_contadores()

    def finalizar_proceso(self):
        self.botonProcesarImagenEntrada.setEnabled(True)

    def mostrar_error(self, error):
        self.visorTexto.setText(error)
        self.botonProcesarImagenEntrada.setEnabled(True)
        
    def handleLimpiar(self):
        self.viewer.clear()
        self.viewer2.clear()
        self.viewer3.clear()
        self.visorTexto.clear()
        self.limpiar_lista_lineas()
        
        self._path = None
        self.OpenCV_image = None
        self.procesedImage = None
        self.img_lineas = None
        self.img_caracteres = None
        
        self.palabra_count = 0
        self.palabra_count_ocr = 0
        self.caracteres_count_ocr = 0
        self.actualizar_contadores()
        
        self.botonProcesarImagenEntrada.setEnabled(False)
        self.botonLimpiar.setEnabled(False)

    def ActualizarPixMap(self, label_target, image):
        if image is None: return
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        qImg = QImage(rgb_image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        
        scaled_pixmap = pixmap.scaled(
            label_target.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        label_target.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        if self.OpenCV_image is not None:
            self.ActualizarPixMap(self.viewer, self.OpenCV_image)
        if hasattr(self, 'img_lineas') and self.img_lineas is not None:
            self.ActualizarPixMap(self.viewer2, self.img_lineas)
        if hasattr(self, 'img_caracteres') and self.img_caracteres is not None:
            self.ActualizarPixMap(self.viewer3, self.img_caracteres)
        super().resizeEvent(event)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())