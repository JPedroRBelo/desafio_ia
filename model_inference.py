import argparse
import logging

from net.resnet50 import SegmentationResnet50
from config.config import IMG_WIDTH, IMG_HEIGHT

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inferência para segmentação de imagens de vegetação a partir de modelo treinado.')
    parser.add_argument('--rgb', required=True, help='Imagem RGB a ser segmentada.')    
    parser.add_argument('--modelpath', required=True, help='Caminho para carregar o modelo.')
    parser.add_argument('--output', required=True, help='Caminho para salvar o arquivo de imagem segmentada.')

    args = parser.parse_args()
    # Configuração do logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    resnet50 = SegmentationResnet50(
        img_width= IMG_WIDTH,
        img_height= IMG_HEIGHT,
        model_path=args.modelpath)

    resnet50.predict(rgb_path=args.rgb, save_path= args.output)
    logging.info("Resultado com imagem segmentada salvo em: %s", args.output)


