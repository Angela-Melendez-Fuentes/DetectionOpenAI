import sys
import cv2
import numpy as np
import base64
import os
from openai import OpenAI
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
# 2. HILO PARA OPENAI
# =========================================================
class WorkerOpenAI(QtCore.QThread):
    linea_procesada = QtCore.pyqtSignal(int, np.ndarray, str)
    proceso_terminado = QtCore.pyqtSignal()
    error_detectado = QtCore.pyqtSignal(str)

    def __init__(self, lista_recortes_lineas):
        super().__init__()
        self.lista_recortes_lineas = lista_recortes_lineas
        self.client = OpenAI() 

    def run(self):
        try:
            for index, img_linea in enumerate(self.lista_recortes_lineas):
                _, buffer = cv2.imencode('.jpg', img_linea)
                base64_image = base64.b64encode(buffer).decode('utf-8')

                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Transcribe el texto manuscrito de esta imagen. Devuelve ÚNICAMENTE el texto transcrito, sin comillas, sin formato markdown y sin ningún comentario adicional."},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                },
                            ],
                        }
                    ],
                    max_tokens=300,
                    temperature=0.1
                )
                
                texto_linea = response.choices[0].message.content.strip()
                self.linea_procesada.emit(index, img_linea, texto_linea)
            
            self.proceso_terminado.emit()
        except Exception as e:
            self.error_detectado.emit(f"Error con la API de OpenAI:\n{str(e)}")

# =========================================================
# 3. INTERFAZ GRÁFICA
# =========================================================
class VisorImagen(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()  
        self.setFrameShape(QtWidgets.QFrame.Shape.Box) 
        self.setLineWidth(2)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumSize(350, 400)

class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector de Escritura a Mano + OpenAI (GPT-4o)")
        self.resize(1200, 750) 
        
        self.OpenCV_image = None  
        self.procesedImage = None 
        self.img_lineas = None
        self.img_caracteres = None
        
        # --- NUEVO: Diccionario para evitar duplicados y mantener orden ---
        self.transcripciones_dict = {}
        
        self.palabra_count = 0    
        self.palabra_count_ocr = 0 
        self.caracteres_count_ocr = 0 
        
        self.crear_widgets()
        self.configurar_layout()
        self.conectar_senales()

    def crear_widgets(self):
        self.botonAbrir = QtWidgets.QPushButton("Abrir Imagen")
        self.botonAbrir.setStyleSheet("background-color: #D5FFCC; color: #4D941E; font-weight: bold; border: 1px solid #4D941E; border-radius: 5px;")

        self.botonProcesarImagenEntrada = QtWidgets.QPushButton("Analizar y Transcribir por Líneas (OpenAI)")
        self.botonProcesarImagenEntrada.setStyleSheet("background-color: #CCEDFF; color: #1E5C94; font-weight: bold; border: 1px solid #1E5C94; border-radius: 5px;")

        self.botonLimpiar = QtWidgets.QPushButton("Limpiar")
        self.botonLimpiar.setStyleSheet("background-color: #ffcccc; color: #cc0000; font-weight: bold; border: 1px solid #cc0000; border-radius: 5px;")
        
        for btn in [self.botonAbrir, self.botonProcesarImagenEntrada, self.botonLimpiar]:
            btn.setMinimumHeight(50) 
            btn.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Weight.Bold))

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
        self.tabs_derecha.setStyleSheet("QTabBar::tab { height: 40px; padding: 0 20px; font-weight: bold; }")

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

        self.tab_transcripcion = QtWidgets.QWidget()
        layout_trans = QtWidgets.QVBoxLayout(self.tab_transcripcion)
        layout_trans.addWidget(self.visorTexto)
        
        self.tab_lineas = QtWidgets.QWidget()
        layout_lineas = QtWidgets.QVBoxLayout(self.tab_lineas)
        layout_lineas.addWidget(self.viewer2)
        
        self.tab_caracteres = QtWidgets.QWidget()
        layout_caracteres = QtWidgets.QVBoxLayout(self.tab_caracteres)
        layout_caracteres.addWidget(self.viewer3)
        layout_caracteres.addWidget(self.label_contador_cv)
        
        self.tabs_derecha.addTab(self.tab_lineas_ocr, "📄 OpenAI: Por Líneas")
        self.tabs_derecha.addTab(self.tab_transcripcion, "📝 OpenAI: Completo")
        self.tabs_derecha.addTab(self.tab_lineas, "⬛ OpenCV: Líneas")
        self.tabs_derecha.addTab(self.tab_caracteres, "🟩 OpenCV: Caracteres")

    def configurar_layout(self):
        layout_principal = QtWidgets.QVBoxLayout(self)
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
            w = self.scroll_vbox.itemAt(i).widget()
            if w: w.setParent(None); w.deleteLater()

    def handleOpen(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Seleccionar Imagen", ".", "Imágenes (*.jpg *.png *.jpeg)")
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.OpenCV_image = img
                self.ActualizarPixMap(self.viewer, self.OpenCV_image)
                self.botonProcesarImagenEntrada.setEnabled(True)
                self.botonLimpiar.setEnabled(True)

    def procesar_todo(self):
        if self.OpenCV_image is None: return
        self.botonProcesarImagenEntrada.setEnabled(False) 
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        self.limpiar_lista_lineas()
        self.visorTexto.clear()
        
        # --- Resetear Diccionario ---
        self.transcripciones_dict = {}
        
        self.palabra_count = 0
        self.palabra_count_ocr = 0
        self.caracteres_count_ocr = 0

        img_res = redimensionar(self.OpenCV_image)
        doc, _ = extraer_documento(img_res)
        self.procesedImage = limpiar_fondo_y_cuadricula(doc) 
        
        self.img_lineas, self.img_caracteres, recortes = self.analizar_lineas_y_caracteres(self.procesedImage, doc)
        
        self.ActualizarPixMap(self.viewer2, self.img_lineas)
        self.ActualizarPixMap(self.viewer3, self.img_caracteres)
        
        self.worker = WorkerOpenAI(recortes)
        self.worker.linea_procesada.connect(self.mostrar_resultado_linea)
        self.worker.proceso_terminado.connect(self.finalizar_proceso)
        self.worker.start()

    def analizar_lineas_y_caracteres(self, img_procesada, img_original):
        gray = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2GRAY)
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        
        img_lineas_visual = img_original.copy()
        img_caracteres_visual = img_original.copy()

        # Kernels
        k_parrafo = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 20))
        k_linea = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        k_palabra = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1))
        k_char = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        mask_p = cv2.dilate(bin_img, k_parrafo, iterations=1)
        cnts_p, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes_p = sorted([cv2.boundingRect(c) for c in cnts_p], key=lambda b: b[1])
        
        lista_recortes = []

        for px, py, pw, ph in bboxes_p:
            if ph < 30 or pw < 30: continue
            roi_p_bin = bin_img[py:py+ph, px:px+pw]
            
            mask_l = cv2.dilate(roi_p_bin, k_linea, iterations=1)
            cnts_l, _ = cv2.findContours(mask_l, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bboxes_l = sorted([cv2.boundingRect(c) for c in cnts_l], key=lambda b: b[1])
            
            for lx, ly, lw, lh in bboxes_l:
                if lh < 10: continue
                gx_l, gy_l = px + lx, py + ly
                roi_l_bin = bin_img[gy_l:gy_l+lh, gx_l:gx_l+lw]

                # --- FILTRO 1: Densidad de píxeles ---
                pixel_density = cv2.countNonZero(roi_l_bin) / (lw * lh)
                if pixel_density < 0.01: continue 

                lista_recortes.append(img_original[gy_l:gy_l+lh, gx_l:gx_l+lw])
                cv2.rectangle(img_lineas_visual, (gx_l, gy_l), (gx_l + lw, gy_l + lh), (255, 0, 0), 2)
                
                # Palabras y Caracteres
                mask_w = cv2.dilate(roi_l_bin, k_palabra, iterations=1)
                cnts_w, _ = cv2.findContours(mask_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for wx, wy, ww, wh in [cv2.boundingRect(c) for c in cnts_w]:
                    if ww < 5 or wh < 5: continue
                    roi_w_bin = roi_l_bin[wy:wy+wh, wx:wx+ww]
                    mask_c = cv2.dilate(roi_w_bin, k_char, iterations=1)
                    cnts_c, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cx, cy, cw, ch in [cv2.boundingRect(c) for c in cnts_c]:
                        if cw > 2 and ch > 8:
                            self.palabra_count += 1
                            cv2.rectangle(img_caracteres_visual, (gx_l+wx+cx, gy_l+wy+cy), (gx_l+wx+cx+cw, gy_l+wy+cy+ch), (0, 200, 0), 1)

        self.actualizar_contadores()
        return img_lineas_visual, img_caracteres_visual, lista_recortes

    def mostrar_resultado_linea(self, index, img_linea, texto):
        # --- FILTRO 2: Ignorar respuestas de error ---
        mensajes_invalidos = ["lo siento", "no puedo", "no hay texto", "imagen proporcionada", "lo lamento"]
        if any(msg in texto.lower() for msg in mensajes_invalidos) or len(texto.strip()) < 1:
            return

        # --- NUEVO: Guardar en el diccionario para evitar duplicados ---
        # Si por alguna razón la señal se repite para el mismo index, no hacemos nada
        if index in self.transcripciones_dict:
            return
            
        self.transcripciones_dict[index] = texto

        # Dibujar en el Scroll (Línea por Línea)
        group_box = QtWidgets.QGroupBox(f"LÍNEA {index + 1}:")
        layout_h = QtWidgets.QHBoxLayout(group_box)
        lbl_img = QtWidgets.QLabel()
        h, w, _ = img_linea.shape
        qImg = QImage(cv2.cvtColor(img_linea, cv2.COLOR_BGR2RGB).data, w, h, 3*w, QImage.Format.Format_RGB888)
        lbl_img.setPixmap(QPixmap.fromImage(qImg).scaledToHeight(50, Qt.TransformationMode.SmoothTransformation))
        
        lbl_texto = QtWidgets.QTextEdit()
        lbl_texto.setReadOnly(True)
        lbl_texto.setText(f"> OpenAI: {texto}")
        lbl_texto.setMaximumHeight(60)
        
        layout_h.addWidget(lbl_img, 1)
        layout_h.addWidget(lbl_texto, 3)
        self.scroll_vbox.addWidget(group_box)
        
        # --- RECONSTRUCCIÓN DINÁMICA DEL TEXTO COMPLETO ---
        # Ordenamos el diccionario por llave (index) y unimos todo con saltos de línea
        texto_completo_ordenado = "\n".join([self.transcripciones_dict[i] for i in sorted(self.transcripciones_dict.keys())])
        self.visorTexto.setText(texto_completo_ordenado)
        
        # Actualizar contadores basándose en el texto real acumulado
        palabras_totales = texto_completo_ordenado.split()
        self.palabra_count_ocr = len(palabras_totales)
        self.caracteres_count_ocr = len(texto_completo_ordenado.replace(" ", "").replace("\n", ""))
        self.actualizar_contadores()

    def finalizar_proceso(self):
        self.botonProcesarImagenEntrada.setEnabled(True)
        QtWidgets.QApplication.restoreOverrideCursor()

    def handleLimpiar(self):
        for v in [self.viewer, self.viewer2, self.viewer3]: v.clear()
        self.visorTexto.clear()
        self.limpiar_lista_lineas()
        self.transcripciones_dict = {} # Limpiar diccionario
        self.palabra_count = self.palabra_count_ocr = self.caracteres_count_ocr = 0
        self.actualizar_contadores()

    def ActualizarPixMap(self, label_target, image):
        if image is None: return
        h, w, _ = image.shape
        qImg = QImage(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).data, w, h, 3*w, QImage.Format.Format_RGB888)
        label_target.setPixmap(QPixmap.fromImage(qImg).scaled(label_target.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())