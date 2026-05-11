import sys
import cv2
import numpy as np
import base64
import os
import re  # Para limpiar puntuación en la comparativa
from openai import OpenAI
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap

# =========================================================
# 0. TEXTO DE REFERENCIA (110 PALABRAS)
# =========================================================
TEXTO_REFERENCIA = (
    "Una vez, cuando tenía seis años, vi un magnífico dibujo en un libro sobre la selva virgen "
    "que se llamaba Historias vividas. Representaba a una serpiente boa tragándose a una fiera. "
    "He aquí la copia del dibujo. En el libro decía: \"Las serpientes boas se tragan sus presas "
    "enteras, sin masticarlas. Después no pueden moverse, y duermen los seis meses que tarda la "
    "digestión. Reflexioné mucho entonces sobre las aventuras de la selva y, a mi vez, logré "
    "trazar con un lápiz de color mi primer dibujo. Mi dibujo número uno era así: Mostré mi "
    "obra maestra a las personas mayores y les pregunté si mi dibujo les daba miedo."
)

# =========================================================
# 1. FUNCIONES DE PREPROCESAMIENTO
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
    return cv2.cvtColor(cv2.bitwise_not(limpia), cv2.COLOR_GRAY2BGR)

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
                
                # PROMPT AJUSTADO PARA EVITAR DISCULPAS
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": "Transcribe el texto manuscrito. Devuelve ÚNICAMENTE las palabras que entiendas. Si no entiendes nada o la imagen es borrosa, no devuelvas NADA, deja la respuesta en blanco. Prohibido decir 'lo siento', 'no puedo' o explicar por qué no puedes."
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }],
                    max_tokens=300,
                    temperature=0
                )
                texto_linea = response.choices[0].message.content.strip()
                self.linea_procesada.emit(index, img_linea, texto_linea)
            self.proceso_terminado.emit()
        except Exception as e:
            self.error_detectado.emit(f"Error de conexión: {str(e)}")

# =========================================================
# 3. INTERFAZ GRÁFICA
# =========================================================
class VisorImagen(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()  
        self.setFrameShape(QtWidgets.QFrame.Shape.Box) 
        self.setLineWidth(2)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(350, 400)

class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analizador de Caligrafía - Comparativa 110 Palabras")
        self.resize(1500, 900)
        self.OpenCV_image = None  
        self.crear_widgets()
        self.configurar_layout()
        self.conectar_senales()

    def crear_widgets(self):
        self.botonAbrir = QtWidgets.QPushButton("Abrir Imagen")
        self.botonProcesarImagenEntrada = QtWidgets.QPushButton("Analizar y Transcribir (OpenAI)")
        self.botonLimpiar = QtWidgets.QPushButton("Limpiar")
        
        self.botonAbrir.setStyleSheet("background-color: #D5FFCC; color: #4D941E; font-weight: bold; border: 1px solid #4D941E; border-radius: 5px;")
        self.botonProcesarImagenEntrada.setStyleSheet("background-color: #CCEDFF; color: #1E5C94; font-weight: bold; border: 1px solid #1E5C94; border-radius: 5px;")
        self.botonLimpiar.setStyleSheet("background-color: #ffcccc; color: #cc0000; font-weight: bold; border: 1px solid #cc0000; border-radius: 5px;")
        
        for btn in [self.botonAbrir, self.botonProcesarImagenEntrada, self.botonLimpiar]:
            btn.setMinimumHeight(50)
            btn.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Weight.Bold))

        self.viewer = VisorImagen()
        self.viewer2 = VisorImagen()
        self.viewer3 = VisorImagen()
        
        self.visorTexto = QtWidgets.QTextEdit()
        self.visorTexto.setReadOnly(True)
        self.visorTexto.setFont(QtGui.QFont("Arial", 14))
        
        self.tabs_derecha = QtWidgets.QTabWidget()
        self.scroll_ocr = QtWidgets.QScrollArea()
        self.scroll_ocr.setWidgetResizable(True)
        self.scroll_widget = QtWidgets.QWidget()
        self.scroll_vbox = QtWidgets.QVBoxLayout(self.scroll_widget)
        self.scroll_vbox.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_ocr.setWidget(self.scroll_widget)
        
        self.tabs_derecha.addTab(self.scroll_ocr, "Líneas Detectadas")
        self.tabs_derecha.addTab(self.visorTexto, "Transcripción y Comparativa")
        self.tabs_derecha.addTab(self.viewer2, "OpenCV: Líneas")
        self.tabs_derecha.addTab(self.viewer3, "OpenCV: Caracteres")

    def configurar_layout(self):
        layout_principal = QtWidgets.QVBoxLayout(self)
        layout_botones = QtWidgets.QHBoxLayout()
        layout_botones.addWidget(self.botonAbrir); layout_botones.addWidget(self.botonProcesarImagenEntrada); layout_botones.addWidget(self.botonLimpiar)
        layout_principal.addLayout(layout_botones)
        
        cuerpo = QtWidgets.QHBoxLayout()
        cuerpo.addWidget(self.viewer, 1)
        cuerpo.addWidget(self.tabs_derecha, 1)
        layout_principal.addLayout(cuerpo)

    def conectar_senales(self):
        self.botonAbrir.clicked.connect(self.handleOpen)
        self.botonProcesarImagenEntrada.clicked.connect(self.procesar_todo)
        self.botonLimpiar.clicked.connect(self.handleLimpiar)

    def handleOpen(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Seleccionar Imagen", ".", "Imágenes (*.jpg *.png *.jpeg)")
        if path:
            self.OpenCV_image = cv2.imread(path)
            self.ActualizarPixMap(self.viewer, self.OpenCV_image)
            self.botonProcesarImagenEntrada.setEnabled(True)

    def procesar_todo(self):
        if self.OpenCV_image is None: return
        self.botonProcesarImagenEntrada.setEnabled(False)
        self.limpiar_lista_lineas()
        self.visorTexto.clear()
        
        img_res = redimensionar(self.OpenCV_image)
        doc, _ = extraer_documento(img_res)
        procesada = limpiar_fondo_y_cuadricula(doc)
        
        self.img_lineas, self.img_caracteres, recortes = self.analizar_lineas_y_caracteres(procesada, doc)
        self.ActualizarPixMap(self.viewer2, self.img_lineas)
        self.ActualizarPixMap(self.viewer3, self.img_caracteres)
        
        self.worker = WorkerOpenAI(recortes)
        self.worker.linea_procesada.connect(self.mostrar_resultado_linea)
        self.worker.proceso_terminado.connect(self.finalizar_proceso)
        self.worker.start()

    def analizar_lineas_y_caracteres(self, img_procesada, img_original):
        gray = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        img_lineas_vis = img_original.copy()
        img_chars_vis = img_original.copy()
        
        kernel_l = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        mask_lineas = cv2.dilate(bin_img, kernel_l, iterations=1)
        contornos, _ = cv2.findContours(mask_lineas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = sorted([cv2.boundingRect(c) for c in contornos if cv2.boundingRect(c)[3] > 10], key=lambda b: b[1])
        recortes = []
        for x, y, w, h in bboxes:
            cv2.rectangle(img_lineas_vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
            recortes.append(img_original[y:y+h, x:x+w])
            
        return img_lineas_vis, img_chars_vis, recortes

    def mostrar_resultado_linea(self, index, img_linea, texto):
        # FILTRO DE SEGURIDAD: Si OpenAI devuelve una frase de error, la ignoramos.
        frases_bloqueadas = ["lo siento", "no puedo", "no es posible", "imagen borrosa", "transcribir el texto"]
        if any(f in texto.lower() for f in frases_bloqueadas):
            texto = "" 

        if texto.strip(): # Solo añadir si hay texto real
            group = QtWidgets.QGroupBox(f"Línea {index+1}")
            lay = QtWidgets.QHBoxLayout(group)
            lbl_img = QtWidgets.QLabel()
            h, w, _ = img_linea.shape
            q_img = QImage(cv2.cvtColor(img_linea, cv2.COLOR_BGR2RGB).data, w, h, 3*w, QImage.Format.Format_RGB888)
            lbl_img.setPixmap(QPixmap.fromImage(q_img).scaledToHeight(50))
            lbl_txt = QtWidgets.QLabel(texto)
            lay.addWidget(lbl_img); lay.addWidget(lbl_txt)
            self.scroll_vbox.addWidget(group)
            
            self.visorTexto.insertPlainText(texto + " ")

    def finalizar_proceso(self):
        texto_final = self.visorTexto.toPlainText().strip()
        
        def limpiar_lista(t):
            t_limpio = re.sub(r'[^\w\s]', '', t.lower())
            return t_limpio.split()

        palabras_ref = limpiar_lista(TEXTO_REFERENCIA)
        palabras_trans = limpiar_lista(texto_final)
        
        coincidencias = 0
        errores = []
        
        # Comparación estricta palabra por palabra
        for i in range(len(palabras_ref)):
            original = palabras_ref[i]
            if i < len(palabras_trans):
                leida = palabras_trans[i]
                if original == leida:
                    coincidencias += 1
                else:
                    errores.append(f"Posición {i+1}: Original '{original}' | Leído '{leida}'")
            else:
                errores.append(f"Posición {i+1}: FALTA la palabra '{original}'")

        reporte = "\n\n" + "="*50 + "\n"
        reporte += "COMPARATIVA DE PRECISIÓN (110 PALABRAS DEL TEXTO ORIGINAL)\n"
        reporte += f"• Palabras en Original: {len(palabras_ref)}\n"
        reporte += f"• Palabras Detectadas: {len(palabras_trans)}\n"
        reporte += f"• Coincidencias Exactas: {coincidencias}\n"
        reporte += f"• Fallos (errores o faltantes): {len(palabras_ref) - coincidencias}\n"
        reporte += "="*50 + "\n"
        
        if errores:
            reporte += "LISTADO DE DIFERENCIAS:\n" + "\n".join(errores[:30])
            if len(errores) > 30: reporte += "\n... (más diferencias ocultas)"
        else:
            reporte += "🎉 ¡PERFECTO! Todas las palabras coinciden exactamente."

        self.visorTexto.append(reporte)
        self.botonProcesarImagenEntrada.setEnabled(True)
        self.botonProcesarImagenEntrada.setText("Analizar y Transcribir (OpenAI)")
        QtWidgets.QApplication.restoreOverrideCursor()

    def limpiar_lista_lineas(self):
        while self.scroll_vbox.count():
            item = self.scroll_vbox.takeAt(0)
            if item.widget(): item.widget().deleteLater()

    def handleLimpiar(self):
        self.viewer.clear(); self.visorTexto.clear(); self.limpiar_lista_lineas()
        self.OpenCV_image = None
        self.botonProcesarImagenEntrada.setEnabled(False)

    def ActualizarPixMap(self, label, image):
        if image is None: return
        h, w, _ = image.shape
        q_img = QImage(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).data, w, h, 3*w, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_img).scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())