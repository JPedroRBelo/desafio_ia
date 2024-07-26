import os
import logging
from PIL import Image
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import BinaryIoU
from tensorflow.keras.callbacks import TensorBoard

class SegmentationResnet50:
    """
    Classe para treinar o modelo DeepLabV3+ com ResNet50 para segmentação semântica de áreas vegetadas.
    """
    def __init__(self, img_width: int = 256, img_height: int = 256, batch_size: int = 8, epochs: int = 100, learning_rate : float = 0.0001, validation_split: float = 0.1, test_split: float = 0, model_path: Optional[str] = None):
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.test_split = test_split
        self.model = self.create_model(input_size=(self.img_height, self.img_width, 3))
        self.logdir = 'logs'
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            self.model = self.create_model(input_size=(self.img_height, self.img_width, 3))

    def create_model(self, input_size: Tuple[int, int, int]) -> Model:
        """
        Cria o modelo DeepLabV3+ com ResNet50 como backbone.

        Args:
            input_size (Tuple): Dimensão de entrada da imagem (altura, largura, canais).

        Returns:
            Model: Modelo compilado.
        """

        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_size)
        
        # Extraindo camadas relevantes para o DeepLabV3+
        x = base_model.output

        # Atrous Spatial Pyramid Pooling (ASPP)
        # ASPP é um módulo de segmentação semântica para reamostragem de uma dada camada
        # de feições em múltiplas taxas antes da convolução. 
        # Isso equivale a sondar a imagem original com múltiplos filtros que têm campos
        # de visão efetivos complementares, capturando assim objetos, bem como contexto
        # de imagem útil em múltiplas escalas. Em vez de realmente reamostragem de feições, 
        # o mapeamento é implementado usando múltiplas camadas convolucionais atrous 
        # paralelas com diferentes taxas de amostragem.

        b4 = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
        b4 = Conv2D(256, (3, 3), padding='same', activation='relu', dilation_rate=(6, 6))(b4)  
        b6 = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
        b6 = Conv2D(256, (3, 3), padding='same', activation='relu', dilation_rate=(12, 12))(b6)    
        b8 = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
        b8 = Conv2D(256, (3, 3), padding='same', activation='relu', dilation_rate=(18, 18))(b8)
    
        # Concatenando e aplicando uma convolução final
        x = Concatenate()([x, b4, b6, b8])
        x = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
        
        # Upsampling para atingir a resolução final
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)

        # UpSampling final
        x = UpSampling2D(size=(2, 2))(x)  
        
        # Camada de saída com a mesma dimensão das máscaras
        x = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
        
        model = Model(inputs=base_model.input, outputs=x)
        model.compile(  SGD(learning_rate=self.learning_rate, weight_decay=0.0001, momentum=0.9, clipnorm=10.0),
                        loss='binary_crossentropy',
                        metrics=[BinaryIoU(threshold=0.5),'accuracy']
        )  
        return model
    
    def load_data(self, img_dir: str, mask_dir: str, img_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carrega e preprocessa as imagens e máscaras usando PIL.

        Args:
            img_dir (str): Diretório contendo as imagens.
            mask_dir (str): Diretório contendo as máscaras.
            img_size (Tuple): Dimensão desejada para as imagens (altura, largura).

        Returns:
            Tuple: Arrays numpy contendo as imagens e máscaras preprocessadas.
        """
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        
        images = []
        masks = []
        
        for img_file, mask_file in zip(img_files, mask_files):
            img = Image.open(os.path.join(img_dir, img_file)).convert('RGB').resize((img_size[1], img_size[0]))
            img = np.array(img) / 255.0  # Normalizar para [0, 1]
            
            mask = Image.open(os.path.join(mask_dir, mask_file)).resize((img_size[1], img_size[0]))
            mask = np.array(mask)
            mask = np.expand_dims(mask, axis=-1)
            
            images.append(img)
            masks.append(mask)
        
        return np.array(images), np.array(masks)

    def plot_results(self,images, masks, predictions, num=5):
        """
        Plota os resultados das previsões do modelo.

        Args:
            images (np.ndarray): Conjunto de imagens originais.
            masks (np.ndarray): Conjunto de máscaras reais.
            predictions (np.ndarray): Conjunto de previsões do modelo.
            num (int): Número de amostras a serem plotadas.
        """
        _, axes = plt.subplots(num, 3, figsize=(15, num * 5))
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

    def save_model(self, model_path: str = os.path.join('models','veg_segmentation.h5')) -> None:
        """
        Salva o modelo.

        Args:
            model_path (str): Caminho para salvar o modelo.
        """
        self.model.save(model_path)

    def train(self, dataset_dir: str, masks_dir: str) -> Model:
        """
        Treina o modelo DeepLabV3+ com ResNet50 usando as imagens e máscaras.

        Args:
            rgb_path (str): Diretório contendo as imagens RGB.
            groundtruth_path (str): Diretório contendo as máscaras de verdade-terreno.
        """
        # Carregar os dados
        images, masks = self.load_data(dataset_dir, masks_dir, (self.img_height, self.img_width))

        # Separar o conjunto de treino e teste
        split_idx = int(len(images) * (1 - self.test_split))
        train_images, test_images = images[:split_idx], images[split_idx:]
        train_masks, test_masks = masks[:split_idx], masks[split_idx:]

        #Configuração do tensorboard
        tensorboard_callback = TensorBoard(log_dir=self.logdir, histogram_freq=1)
        # Treinamento do modelo
        self.model.fit(train_images,
                       train_masks,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_split=self.validation_split,
                       callbacks=[tensorboard_callback]
        )

        if len(test_images) > 0:
            # Predições no conjunto de teste
            predictions = self.model.predict(test_images)
            # Arredonda as segmentações para garantir que sejam binárias
            predictions = (predictions > 0.5).astype(np.uint8)

            # Plota os resultados
            self.plot_results(test_images, test_masks, predictions)
        return self.model
    

    def predict(self, rgb_path: str, save_path: str) -> Model:
        """
        Segmenta imagem a partir de modelo.

        Args:
            rgb_path (str): Diretório contendo as imagens RGB.
            save_path (str): Caminho para salvar resultado.
        """
        img = Image.open(rgb_path).convert('RGB')#.resize((img_size[1], img_size[0]))
        #img = np.array(img) / 255.0  # Normalizar para [0, 1]

        img_width, img_height = img.size
        result_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        for i in range(0, img_width, self.img_width):
            for j in range(0, img_height, self.img_height):
                #Calcula box a ser recortado da imagem de origem
                box = (i, j, i + self.img_width, j + self.img_height)
                tile = img.crop(box)
                tile = np.array(tile) / 255.0  # Normalizar para [0, 1]
                tile = np.expand_dims(tile, axis=0)
                prediction = self.model.predict(tile)
                # Arredonda as segmentações para garantir que sejam binárias
                prediction = (prediction > 0.5).astype(np.uint8)
                # Remover a dimensão do batch e do canal
                current_tile_width = min(self.img_width, img_width - i)
                
                current_tile_height = min(self.img_height, img_height - j)
                print(current_tile_height)
                prediction = prediction[0, :current_tile_height, :current_tile_width, 0]
                # Constroi a imagem de saída
                result_mask[j:j + current_tile_height, i:i + current_tile_width] = prediction



        result_image = Image.fromarray(result_mask * 255)
        result_image.save(save_path)