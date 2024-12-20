import os
import shutil
from random import shuffle

# Rutas de los archivos y carpetas
images_dir = 'dataset/imgs'  # Ruta a la carpeta con las imágenes
output_base_dir = 'dataset/organized'  # Carpeta base para las imágenes organizadas

# Crear la carpeta base si no existe
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# Obtener los nombres de todas las imágenes en la carpeta
imagenes = [img for img in os.listdir(images_dir) if img.endswith('.jpg') or img.endswith('.png')]

# Verificar el total de imágenes
total_imagenes = len(imagenes)
print(f"Total de imágenes encontradas: {total_imagenes}")

# Definir cantidades exactas para entrenamiento, validación y prueba
train_count = 10297  # 70% de 14,710
val_count = 2207     # 15% de 14,710
test_count = 2206    # 15% de 14,710

# Comprobar que la suma de imágenes coincide con el total
if train_count + val_count + test_count != total_imagenes:
    raise ValueError("La suma de las imágenes de entrenamiento, validación y prueba no coincide con el total de imágenes.")

# Mezclar las imágenes aleatoriamente
shuffle(imagenes)

# Dividir las imágenes en los conjuntos de entrenamiento, validación y prueba
train_data = imagenes[:train_count]
val_data = imagenes[train_count:train_count + val_count]
test_data = imagenes[train_count + val_count:]

# Función para mover imágenes a carpetas específicas
def copy_images(imagenes_list, source_dir, dest_dir):
    for image_name in imagenes_list:
        source_path = os.path.join(source_dir, image_name)
        estado_maduracion = image_name.split('_')[-1].split('.')[0]  # Extraer estado de maduración
        estado_dir = os.path.join(dest_dir, estado_maduracion)
        
        if not os.path.exists(estado_dir):
            os.makedirs(estado_dir)
            
        dest_path = os.path.join(estado_dir, image_name)
        
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)
        else:
            print(f"Advertencia: {image_name} no se encontró en {source_dir}")

# Mover imágenes a las carpetas de entrenamiento, validación y prueba
copy_images(train_data, images_dir, os.path.join(output_base_dir, 'train'))
copy_images(val_data, images_dir, os.path.join(output_base_dir, 'validation'))
copy_images(test_data, images_dir, os.path.join(output_base_dir, 'test'))

print("Las imágenes han sido organizadas en carpetas de entrenamiento (10,297), validación (2,207) y prueba (2,206).")
# Imprimir la cantidad de imágenes en cada conjunto
print(f"Cantidad de imágenes en entrenamiento: {len(train_data)}")
print(f"Cantidad de imágenes en validación: {len(val_data)}")
print(f"Cantidad de imágenes en prueba: {len(test_data)}")
