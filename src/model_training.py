##Codigo de entrenamiento en Google Colab
from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers, regularizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuración de directorios
train_dir = '/content/drive/MyDrive/organized/train'
validation_dir = '/content/drive/MyDrive/organized/validation'
test_dir = '/content/drive/MyDrive/organized/test'


# Configuración de hiperparámetros
img_height = 224
img_width = 224
batch_size = 64
initial_lr = 1e-3
max_epochs = 50

# Generadores de datos
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Calcular pesos de las clases
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))


# Modelo base InceptionV3
base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Congelar las primeras 100 capas
for layer in base_model.layers[:100]:
    layer.trainable = False

# Construcción del modelo completo
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Dropout(0.6)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(train_generator.num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# Compilación del modelo
model.compile(optimizer=optimizers.Adam(learning_rate=initial_lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
class StopAtAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, target_accuracy): 
        super(StopAtAccuracy, self).__init__()
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy')
        if accuracy and accuracy >= self.target_accuracy:
            print(f"\n¡Meta alcanzada! Precisión del {accuracy*100:.2f}% en la época {epoch+1}")
            self.model.stop_training = True

reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1)
stop_at_95 = StopAtAccuracy(target_accuracy=0.95)
model_checkpoint = callbacks.ModelCheckpoint(
    filepath='/content/drive/MyDrive/resultados/mejor_modelo.keras',  # Guardar en Google Drive
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Crear carpeta en Google Drive
os.makedirs('/content/drive/MyDrive/resultados', exist_ok=True)

# Entrenamiento del modelo
history = model.fit(
    train_generator,
    epochs=max_epochs,
    validation_data=validation_generator,
    class_weight=class_weights_dict,
    callbacks=[reduce_lr, stop_at_95, model_checkpoint],
    verbose=1
)

# Guardar modelo final
model.save('/content/drive/MyDrive/resultados/modelo_optimo_final.keras')

# Graficar precisión y pérdida
plt.figure(figsize=(12, 5))

# Gráfica de precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión en Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión en Validación')
plt.title('Precisión por Época')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.grid()

# Gráfica de pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida en Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida en Validación')
plt.title('Pérdida por Época')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Evaluación del modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"\nPrecisión en datos de prueba: {test_accuracy*100:.2f}%")

# Predicciones y matriz de confusión
y_true = test_generator.classes
y_pred = np.argmax(model.predict(test_generator), axis=1)

# Reporte de clasificación
print("\nReporte de Clasificación:")
target_names = list(test_generator.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=target_names))

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title('Matriz de Confusión')
plt.show()