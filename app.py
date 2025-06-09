import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import os # Necesario para verificar si el archivo de modelo existe

# --- Configuración de la aplicación Streamlit ---
st.set_page_config(layout="wide")
st.title("Clasificador de Mazorcas de Cacao 🍫")
st.write("Sube una imagen de una mazorca de cacao para predecir su estado de madurez (verde/madura/enferma).")

# --- Definir parámetros globales (igual que en tus bloques) ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 3 # 'verde', 'madura', 'enferma'
class_names = ['enferma', 'madura', 'verde'] # ¡ASEGÚRATE DE QUE ESTE ORDEN COINCIDE CON EL ENTRENAMIENTO!
                                          # Revisa el Bloque 2: `train_generator.class_indices.keys()`

# --- Carga del modelo ---
# La ruta del modelo debe ser RELATIVA a la raíz de tu proyecto Streamlit
# Si tu modelo está en la misma carpeta que app.py:
MODEL_PATH = './models/cacao_resnet101_classifier.keras'
# Si tu modelo está en una subcarpeta llamada 'modelos':
# MODEL_PATH = "modelos/cacao_resnet101_classifier.keras"


@st.cache_resource # Carga el modelo una sola vez para mejorar el rendimiento
def load_cacao_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: El archivo del modelo '{MODEL_PATH}' no se encontró. "
                 "Asegúrate de que el modelo esté en la raíz de tu repositorio de GitHub "
                 "o en la subcarpeta correcta que especificaste.")
        st.stop() # Detiene la ejecución si el modelo no se encuentra
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo. Esto puede ser por una versión incompatible de TensorFlow o un archivo dañado: {e}")
        st.stop()

model = load_cacao_model()

# --- Funciones de preprocesamiento (Adaptadas de Bloque 8) ---

def preprocess_image_for_resnet(image_pil):
    """
    Preprocesa una imagen PIL para la entrada de ResNet.
    Ajusta el tamaño y aplica la función de preprocesamiento de ResNet.
    """
    img_array = np.array(image_pil.convert('RGB')).astype(np.float32) # Asegurar RGB y float32
    img_resized = Image.fromarray(img_array.astype(np.uint8)).resize((IMG_WIDTH, IMG_HEIGHT))
    img_array_resized = np.asarray(img_resized, dtype=np.float32)
    img_preprocessed = tf.keras.applications.resnet.preprocess_input(np.expand_dims(img_array_resized, axis=0))
    return img_preprocessed

def extract_numerical_features_for_prediction(image_pil):
    """
    Extrae características numéricas como porcentaje de píxeles negros,
    conteo de manchas negras grandes y promedio de valor verde de una imagen PIL.
    """
    img_bgr = cv2.cvtColor(np.array(image_pil.convert('RGB')), cv2.COLOR_RGB2BGR) # Convertir PIL a BGR para OpenCV

    # 1. Porcentaje de píxeles negros
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, black_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    percentage_black = (np.sum(black_mask > 0) / (img_bgr.shape[0] * img_bgr.shape[1])) * 100

    # 2. Conteo de manchas negras grandes
    num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(black_mask, 8, cv2.CV_32S)
    large_black_spots_count = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 100: # <--- Ajusta este umbral de área
            large_black_spots_count += 1
            
    # 3. Promedio de valor verde (en espacio de color HSV)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    mean_green_value = 0
    if np.sum(green_mask > 0) > 0:
        mean_green_value = np.mean(hsv[green_mask > 0, 2])

    # Normalizar las características (igual que en el entrenamiento)
    max_percentage_black = 100.0
    max_spots = 20
    max_green_value = 255.0

    scaled_percentage_black = percentage_black / max_percentage_black
    scaled_large_black_spots_count = min(large_black_spots_count / max_spots, 1.0)
    scaled_mean_green_value = mean_green_value / max_green_value

    return np.array([scaled_percentage_black, scaled_large_black_spots_count, scaled_mean_green_value], dtype=np.float32)


# --- Interfaz de usuario de Streamlit ---
uploaded_file = st.file_uploader("Elige una imagen de mazorca de cacao...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida', use_column_width=True)
    st.write("")

    if st.button("Clasificar Mazorca"):
        with st.spinner('Clasificando la imagen...'):
            try:
                # Preprocesar imagen para ResNet
                preprocessed_img_for_resnet = preprocess_image_for_resnet(image)

                # Extraer características numéricas
                numerical_features = extract_numerical_features_for_prediction(image)
                numerical_features_batch = np.expand_dims(numerical_features, axis=0) # Añadir dimensión de lote

                # Realizar la predicción con ambas entradas
                predictions = model.predict([preprocessed_img_for_resnet, numerical_features_batch])
                
                # Obtener la clase predicha y su confianza
                predicted_class_idx = np.argmax(predictions, axis=1)[0]
                predicted_class_name = class_names[predicted_class_idx]
                confidence = np.max(predictions) * 100

                st.subheader("Resultado de la Clasificación:")
                st.success(f"La mazorca es: **{predicted_class_name}**")
                st.info(f"Confianza: {confidence:.2f}%")

                st.markdown("---")
                st.subheader("Características Numéricas Extraídas:")
                st.write(f"- Porcentaje de píxeles negros: {numerical_features[0]*100:.2f}%")
                st.write(f"- Conteo de manchas negras grandes: {numerical_features[1]*20:.2f} (estimado)") # Desnormaliza para mostrar
                st.write(f"- Promedio de valor verde (HSV): {numerical_features[2]*255:.2f} (estimado)") # Desnormaliza para mostrar

                st.markdown("---")
                st.subheader("Probabilidades por Clase:")
                for i, class_name_display in enumerate(class_names):
                    st.write(f"- {class_name_display}: {predictions[0][i]*100:.2f}%")

            except Exception as e:
                st.error(f"Ocurrió un error durante la clasificación: {e}")
                st.warning("Asegúrate de que la imagen sea válida y que tu modelo cargue correctamente.")