import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

# Directorios de train y test
train_dir = '../fotos/cifar10/train/'
test_dir =  '../fotos/cifar10/test/'

# Normalizar datos
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Obtener datos de entrenamiento
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),  # CIFAR-10 images are 32x32
    batch_size=32,
    class_mode='categorical'  # Use 'categorical' for multi-class classification
)

# Obtener datos de test
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical'
)

# Número de clases (clasificación multiclase)
num_classes = train_generator.num_classes

print("num_classes: " + str(num_classes) + " \n")

# Printing the number of training and test images
print(f'Número de imágenes en entrenamiento: {train_generator.samples}')
print(f'Número de imágenes en test: {test_generator.samples}')

# Crear arquitectura del modelo -----------------------------------------
model = Sequential()

model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(learning_rate=0.01),
    metrics=['accuracy']
)

# Entrenar el modelo ------------------------------------------------------
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)

# Evaluar el modelo en test -------------------------------------------------
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print('Test accuracy:', test_accuracy)
