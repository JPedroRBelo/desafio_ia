
import argparse
import logging
import numpy as np

from net.resnet50 import SegmentationResnet50
from config.config import IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, EPOCHS, TEST_SPLIT, VAL_SPLIT, LEARNING_RATE

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Treinamento de modelo DeepLabV3+ com ResNet50 para segmentação de vegetação.')
    parser.add_argument('--rgb', required=True, help='Diretório contendo imagens RGB.')
    parser.add_argument('--groundtruth', required=True, help='Diretório contendo máscaras binarizadas.')
    parser.add_argument('--modelpath', required=True, help='Caminho para salvar o modelo final.')

    args = parser.parse_args()
    # Configuração do logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    resnet50 = SegmentationResnet50(
        img_width= IMG_WIDTH,
        img_height= IMG_HEIGHT,
        batch_size= BATCH_SIZE,
        epochs= EPOCHS,
        learning_rate=LEARNING_RATE,
        validation_split= VAL_SPLIT,
        test_split= TEST_SPLIT
    )

    resnet50.train(dataset_dir= args.rgb, masks_dir= args.groundtruth)
    resnet50.save_model(model_path=args.modelpath)
    logging.info("Modelo treinado e salvo em: %s", args.modelpath)


