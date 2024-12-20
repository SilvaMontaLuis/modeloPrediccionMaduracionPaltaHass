##Segundo Entrenamiento del modelo ateriormente entrenado  en colab
# Cargar el modelo guardado
modelo_guardado = load_model('/content/drive/MyDrive/resultados/modelo_optimo_final.keras')

# Compilar el modelo (en caso de que no se haya compilado previamente)
modelo_guardado.compile(
    optimizer=Adam(learning_rate=1e-4),  # Puedes ajustar el learning rate si es necesario
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continuar el entrenamiento por 10 épocas más
history = modelo_guardado.fit(
    train_generator,
    epochs=10,  # 10 épocas adicionales
    validation_data=validation_generator,
    callbacks=[
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True)
    ],
    verbose=1
)

# Guardar el modelo después de las 10 épocas adicionales
modelo_guardado.save('/content/drive/MyDrive/resultados/mejor_modelo_actualizado_10epocas.keras')

# Evaluar el modelo nuevamente después de las 10 épocas adicionales
loss, accuracy = modelo_guardado.evaluate(validation_generator, verbose=1)
print(f"Precisión del modelo después de 10 épocas adicionales: {accuracy*100:.2f}%")