import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Parâmetros
IMG_WIDTH = 256
IMG_HEIGHT = 256
BATCH_SIZE = 8
EPOCHS = 200
DATASET_DIR = 'dataset'
MASKS_DIR = 'masks'
TEST_SPLIT = 0.1

# Função para criar o modelo DeepLabV3+ com ResNet50 como backbone
def deeplabv3_plus_resnet50(input_size=(IMG_HEIGHT, IMG_WIDTH, 3)):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_size)
    
    # Extraindo camadas relevantes para o DeepLabV3+
    x = base_model.output
    
    # Atrous Spatial Pyramid Pooling (ASPP)
    b4 = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    b4 = Conv2D(256, (3, 3), padding='same', activation='relu', dilation_rate=(6, 6))(b4)
    
    b6 = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    b6 = Conv2D(256, (3, 3), padding='same', activation='relu', dilation_rate=(12, 12))(b6)
    
    b8 = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    b8 = Conv2D(256, (3, 3), padding='same', activation='relu', dilation_rate=(18, 18))(b8)
    
    # Concatenar e aplicar uma convolução final
    x = Concatenate()([x, b4, b6, b8])
    x = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    
    # Adicionando camadas de upsampling para atingir a resolução desejada
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)  # UpSampling final
    
    # Camada de saída com a mesma dimensão das máscaras
    x = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    
    # Compilar o modelo com BinaryIoU como métrica
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryIoU(threshold=0.5),'accuracy'])
    
    return model

# Função para carregar e preprocessar as imagens
def load_data(img_dir, mask_dir, img_size):
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
    
    images = []
    masks = []
    
    for img_file, mask_file in zip(img_files, mask_files):
        img = cv2.imread(os.path.join(img_dir, img_file))
        img = cv2.resize(img, (img_size[1], img_size[0]))
        img = img / 255.0  # Normalizar para [0, 1]
        
        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (img_size[1], img_size[0]))
        mask = np.expand_dims(mask, axis=-1)  # Adicionar uma dimensão para o canal
        
        images.append(img)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

# Carregar os dados
images, masks = load_data(DATASET_DIR, MASKS_DIR, (IMG_HEIGHT, IMG_WIDTH))

# Separar o conjunto de treino e teste
split_idx = int(len(images) * (1 - TEST_SPLIT))
train_images, test_images = images[:split_idx], images[split_idx:]
train_masks, test_masks = masks[:split_idx], masks[split_idx:]

# Criar o modelo
model = deeplabv3_plus_resnet50(input_size=(IMG_HEIGHT, IMG_WIDTH, 3))

# Treinar o modelo
model.fit(train_images, train_masks, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)

# Prever no conjunto de teste
predictions = model.predict(test_images)

# Arredondar as previsões para garantir que sejam binárias
predictions = (predictions > 0.5).astype(np.uint8)

# Função para plotar os resultados
def plot_results(images, masks, predictions, num=5):
    fig, axes = plt.subplots(num, 3, figsize=(15, num * 5))
    for i in range(num):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title('Imagem Original')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(masks[i].squeeze(), cmap='gray')
        axes[i, 1].set_title('Máscara Real')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(predictions[i].squeeze(), cmap='gray')
        axes[i, 2].set_title('Previsão do Modelo')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Plotar os resultados para algumas amostras do conjunto de teste
plot_results(test_images, test_masks, predictions)
