import sys
import cv2
import numpy as np
import base64
import os
import re
import csv
from openai import OpenAI
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

# =========================================================
# 1. FUNCIONES DE PREPROCESAMIENTO Y VISIÓN (OpenCV Puro)
# =========================================================
def preprocesar_imagen(img):
    """Limpia la imagen unificando la iluminación antes de binarizar."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. CORRECCIÓN DE ILUMINACIÓN
    # Difuminamos la imagen exageradamente para aislar el fondo y sus sombras
    fondo_sombras = cv2.GaussianBlur(gray, (101, 101), 0)
    
    # Restamos las sombras a la imagen original y la aclaramos
    sin_sombras = cv2.addWeighted(gray, 1, fondo_sombras, -1, 255)
    
    # 2. Binarizamos sobre la imagen perfectamente iluminada
    blur = cv2.GaussianBlur(sin_sombras, (5, 5), 0)
    bin_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    
    # 3. Eliminar líneas de cuaderno
    kernel_lineas = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    lineas = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_lineas, iterations=2)
    sin_lineas = cv2.subtract(bin_img, lineas)
    
    # 4. Limpiar polvillo suelto
    kernel_ruido = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    limpia = cv2.morphologyEx(sin_lineas, cv2.MORPH_OPEN, kernel_ruido, iterations=1)
    
    # 5. Apagar los márgenes oscuros y espirales
    h, w = limpia.shape
    margen_x = int(w * 0.08)
    margen_y = int(h * 0.05)
    limpia[:, :margen_x] = 0        
    limpia[:, w-margen_x:] = 0      
    limpia[:margen_y, :] = 0        
    limpia[h-margen_y:, :] = 0      
    
    return limpia

def detectar_cajas_palabras(img_bgr, bin_img):
    """Detecta las cajas usando un kernel que respeta los renglones."""
    img_cajas = img_bgr.copy()
    
    # Kernel ancho horizontalmente (25) y corto verticalmente (3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    dilated = cv2.dilate(bin_img, kernel, iterations=2)
    
    contornos, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cajas = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        
        # Filtros geométricos para texto lógico
        if 10 < h < 80 and w > 20:
            cajas.append((x, y, w, h))
            cv2.rectangle(img_cajas, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
    # Ordenar agrupando por renglones (aprox 30px)
    cajas = sorted(cajas, key=lambda b: (b[1] // 30, b[0]))
    
    return img_cajas, cajas

def conv_img_base64(img_bgr):
    _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')

# =========================================================
# 2. HILO DE PROCESAMIENTO (OPENAI + COMPARACIÓN ESTRUCTURAL)
# =========================================================
class WorkerAnalisis(QThread):
    progreso = pyqtSignal(str)
    terminado = pyqtSignal(str, str, str) 
    
    def __init__(self, imagen, texto_referencia, api_key):
        super().__init__()
        self.imagen = imagen
        self.texto_referencia = texto_referencia
        self.api_key = api_key

    def run(self):
        try:
            self.progreso.emit("Conectando con OpenAI (GPT-4o)...")
            client = OpenAI(api_key=self.api_key)
            b64_img = conv_img_base64(self.imagen)
            
            respuesta = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transcribe literalmente el texto manuscrito de la imagen, letra por letra. Mantén cualquier error ortográfico, letra omitida o palabra mal formada exactamente como la percibas. Prohibido corregir la gramática o la ortografía. Devuelve ÚNICAMENTE el texto transcrito, sin comentarios adicionales."
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }],
                max_tokens=500,
                temperature=0.0
            )
            
            texto_transcrito = respuesta.choices[0].message.content.strip()
            self.progreso.emit("Generando alineación estructural...")
            
            reporte = self.generar_comparativa(self.texto_referencia, texto_transcrito)
            self.terminado.emit(texto_transcrito, reporte, "")
            
        except Exception as e:
            self.terminado.emit("", "", str(e))

    def generar_comparativa(self, ref, trans):
        def limpiar(t): return re.sub(r'[^\w\s]', '', t).split()
        
        palabras_ref = limpiar(ref)
        palabras_trans = limpiar(trans)
        
        len_ref = len(palabras_ref)
        len_trans = len(palabras_trans)
        
        rep = "="*50 + "\n"
        rep += "ANÁLISIS DE CANTIDAD Y CONCORDANCIA\n"
        rep += "="*50 + "\n"
        rep += f"Palabras esperadas (Referencia): {len_ref}\n"
        rep += f"Palabras detectadas (Manuscrito): {len_trans}\n\n"
        
        if len_ref == len_trans:
            rep += "✅ ¡Excelente! La cantidad de palabras coincide exactamente.\n\n"
        else:
            rep += f"⚠️ Discrepancia en cantidad: diferencia de {abs(len_ref - len_trans)} palabra(s).\n\n"
            
        rep += "ALINEACIÓN ESTRUCTURAL (SECUENCIA):\n"
        rep += "-"*50 + "\n"
        
        max_len = max(len_ref, len_trans)
        for i in range(max_len):
            p_ref = palabras_ref[i] if i < len_ref else "[NO EXISTE EN REFERENCIA]"
            p_trans = palabras_trans[i] if i < len_trans else "[OMITIDA POR EL USUARIO]"
            rep += f"Posición {(i+1):02d} | Ref: {p_ref:<15} -> Escrito: {p_trans}\n"
            
        return rep

# =========================================================
# 3. INTERFAZ GRÁFICA UNIFICADA CON PESTAÑAS Y CSV
# =========================================================
class VisorImagen(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.setLineWidth(2)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 400)

class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Evaluador de Caligrafía - Alineación Estructural")
        self.resize(1400, 850)
        
        self.OpenCV_image = None
        self.texto_referencia = ""
        self.cajas_detectadas = []
        self.api_key = os.environ.get("OPENAI_API_KEY", "") 
        
        self.crear_widgets()
        self.configurar_layout()
        self.conectar_senales()

    def crear_widgets(self):
        estilo_btn = "font-weight: bold; border-radius: 5px; font-size: 14px; min-height: 45px;"
        
        self.btnCargarTxt = QtWidgets.QPushButton("1. Cargar Referencia (.txt)")
        self.btnCargarTxt.setStyleSheet(f"background-color: #FFF2CC; color: #B38600; border: 1px solid #B38600; {estilo_btn}")
        
        self.btnCargarImg = QtWidgets.QPushButton("2. Cargar Manuscrito (.jpg)")
        self.btnCargarImg.setStyleSheet(f"background-color: #D5FFCC; color: #4D941E; border: 1px solid #4D941E; {estilo_btn}")
        
        self.btnAnalizar = QtWidgets.QPushButton("3. Analizar y Mapear")
        self.btnAnalizar.setStyleSheet(f"background-color: #CCEDFF; color: #1E5C94; border: 1px solid #1E5C94; {estilo_btn}")
        self.btnAnalizar.setEnabled(False)
        
        self.btnExportarCSV = QtWidgets.QPushButton("4. Exportar Métricas (CSV)")
        self.btnExportarCSV.setStyleSheet(f"background-color: #E6E6FA; color: #4B0082; border: 1px solid #4B0082; {estilo_btn}")
        self.btnExportarCSV.setEnabled(False)
        
        self.tabs_imagenes = QtWidgets.QTabWidget()
        self.tabs_imagenes.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Weight.Bold))
        
        self.viewer_orig = VisorImagen()
        self.viewer_bin = VisorImagen()
        self.viewer_proc = VisorImagen()
        
        self.tabs_imagenes.addTab(self.viewer_orig, "1. Imagen Original")
        self.tabs_imagenes.addTab(self.viewer_bin, "2. Binarización Sin Sombras")
        self.tabs_imagenes.addTab(self.viewer_proc, "3. Cajas de Palabras")
        
        self.visorReporte = QtWidgets.QTextEdit()
        self.visorReporte.setReadOnly(True)
        self.visorReporte.setFont(QtGui.QFont("Consolas", 12))
        self.visorReporte.setStyleSheet("background-color: #F8F9FA; border: 1px solid #ccc;")
        
        self.lbl_estado = QtWidgets.QLabel("Esperando archivos...")
        self.lbl_estado.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Weight.Bold))

    def configurar_layout(self):
        layout_principal = QtWidgets.QVBoxLayout(self)
        
        layout_botones = QtWidgets.QHBoxLayout()
        layout_botones.addWidget(self.btnCargarTxt)
        layout_botones.addWidget(self.btnCargarImg)
        layout_botones.addWidget(self.btnAnalizar)
        layout_botones.addWidget(self.btnExportarCSV)
        layout_principal.addLayout(layout_botones)
        
        layout_centro = QtWidgets.QHBoxLayout()
        layout_centro.addWidget(self.tabs_imagenes, 1)
        
        layout_texto = QtWidgets.QVBoxLayout()
        lbl_res = QtWidgets.QLabel("Resultados de la Alineación:")
        lbl_res.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Weight.Bold))
        layout_texto.addWidget(lbl_res)
        layout_texto.addWidget(self.visorReporte)
        
        layout_centro.addLayout(layout_texto, 1)
        
        layout_principal.addLayout(layout_centro)
        layout_principal.addWidget(self.lbl_estado)

    def conectar_senales(self):
        self.btnCargarTxt.clicked.connect(self.cargar_txt)
        self.btnCargarImg.clicked.connect(self.cargar_img)
        self.btnAnalizar.clicked.connect(self.iniciar_analisis)
        self.btnExportarCSV.clicked.connect(self.exportar_metricas_csv)

    def cargar_txt(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Seleccionar Texto de Referencia", ".", "Text Files (*.txt)")
        if path:
            with open(path, 'r', encoding='utf-8') as f:
                self.texto_referencia = f.read().strip()
            self.visorReporte.setText(f"[TEXTO DE REFERENCIA CARGADO]\n{self.texto_referencia}\n\nEsperando imagen...")
            self.validar_ejecucion()

    def cargar_img(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Seleccionar Imagen Manuscrita", ".", "Images (*.jpg *.png *.jpeg)")
        if path:
            self.OpenCV_image = cv2.imread(path)
            self.actualizar_pixmap(self.viewer_orig, self.OpenCV_image)
            
            bin_img = preprocesar_imagen(self.OpenCV_image)
            bin_rgb = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
            self.actualizar_pixmap(self.viewer_bin, bin_rgb)
            
            img_cajas, self.cajas_detectadas = detectar_cajas_palabras(self.OpenCV_image, bin_img)
            self.actualizar_pixmap(self.viewer_proc, img_cajas)
            
            self.lbl_estado.setText(f"Imagen cargada. {len(self.cajas_detectadas)} posibles palabras detectadas por OpenCV.")
            
            self.tabs_imagenes.setCurrentIndex(2)
            self.btnExportarCSV.setEnabled(True) 
            self.validar_ejecucion()

    def exportar_metricas_csv(self):
        if not self.cajas_detectadas:
            QtWidgets.QMessageBox.warning(self, "Aviso", "No hay cajas detectadas para exportar.")
            return
            
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Guardar Métricas Topológicas", "metricas_cajas.csv", "CSV Files (*.csv)")
        
        if path:
            try:
                with open(path, 'w', newline='', encoding='utf-8') as archivo_csv:
                    writer = csv.writer(archivo_csv)
                    writer.writerow(["ID_Palabra", "X_Posicion", "Y_Posicion", "Anchura", "Altura", "Area_Total", "Relacion_Aspecto"])
                    
                    for i, (x, y, w, h) in enumerate(self.cajas_detectadas):
                        area = w * h
                        relacion_aspecto = round(w / h, 3)
                        writer.writerow([i+1, x, y, w, h, area, relacion_aspecto])
                        
                self.lbl_estado.setText(f"¡Éxito! Métricas exportadas correctamente a {os.path.basename(path)}")
                QtWidgets.QMessageBox.information(self, "Exportación Exitosa", "Las métricas de las cajas se han guardado en el archivo CSV.")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error al Exportar", f"Hubo un problema al guardar el archivo:\n{e}")

    def validar_ejecucion(self):
        if self.texto_referencia and self.OpenCV_image is not None:
            self.btnAnalizar.setEnabled(True)

    def iniciar_analisis(self):
        if not self.api_key:
            QtWidgets.QMessageBox.warning(self, "Error", "Falta la OPENAI_API_KEY en las variables de entorno.")
            return
            
        self.btnAnalizar.setEnabled(False)
        self.visorReporte.append("\nIniciando análisis estructural...")
        
        self.worker = WorkerAnalisis(self.OpenCV_image, self.texto_referencia, self.api_key)
        self.worker.progreso.connect(self.lbl_estado.setText)
        self.worker.terminado.connect(self.finalizar_analisis)
        self.worker.start()

    def finalizar_analisis(self, transcripcion, reporte, error):
        self.btnAnalizar.setEnabled(True)
        if error:
            self.lbl_estado.setText("Error en el análisis.")
            self.visorReporte.append(f"\n[ERROR]: {error}")
        else:
            self.lbl_estado.setText("Alineación completada.")
            texto_final = f"--- TRANSCRIPCIÓN GPT-4o ---\n{transcripcion}\n\n{reporte}"
            self.visorReporte.setText(texto_final)

    def actualizar_pixmap(self, label, image):
        if image is None: return
        h, w, _ = image.shape
        q_img = QImage(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).data, w, h, 3*w, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_img).scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        if self.OpenCV_image is not None:
            self.actualizar_pixmap(self.viewer_orig, self.OpenCV_image)
            bin_img = preprocesar_imagen(self.OpenCV_image)
            self.actualizar_pixmap(self.viewer_bin, cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR))
            img_cajas, _ = detectar_cajas_palabras(self.OpenCV_image, bin_img)
            self.actualizar_pixmap(self.viewer_proc, img_cajas)
        super().resizeEvent(event)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())