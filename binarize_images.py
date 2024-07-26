import argparse
import os
import logging
from PIL import Image
import numpy as np
from tqdm import tqdm
from config.config import BINARIZE_THRESHOLD, BINARY_MAX

def ExG(image: Image.Image) -> Image.Image:
    """
    Calcula o índice Excess Green (ExG).
    O Excess Green Index (ExG) é calculado como `2 * green - red - blue`.
    Args:
        image (Image.Image): Imagem RGB para binarização.

    Returns:
        Image.Image: Imagem binarizada em escala de cinza, representando o índice ExG.
    """

    img_array = np.array(image)
    # Separa canais RGB
    red, green, blue = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    # Excess Green Index
    index_exg =  2*green - red - blue
    # Converte o array NumPy para uma imagem em escala de cinza
    return Image.fromarray(index_exg)

def binarize(image: Image.Image) -> Image.Image:
    """
    Binariza uma imagem em escala de cinza com base em um limiar.

    Args:
        image (Image.Image): Imagem em escala de cinza a ser binarizada.

    Returns:
        Image.Image: Imagem binária resultante.
    """

    image_array = np.array(image)
    #Considera somente valores não nulos (área transparente da imagem) e valores acima de um dado limiar
    binary_image_array = (image_array > 0) & (image_array < BINARIZE_THRESHOLD)
    #Converte máscara binária para pixels numerados
    binary_image_array = binary_image_array.astype(np.uint8) * BINARY_MAX
    # Reconverte para imagem
    binary_image = Image.fromarray(binary_image_array, mode='L')

    return binary_image

def process_images(input_dir: str, output_dir: str) -> None:
    """
    Processa todas as imagens RGB em dado diretório e gera máscaras binarizadas.

    Args:
        input_dir (str): Diretório contendo as imagens RGB.
        output_dir (str): Diretório onde as imagens binarizadas serão salvas.

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #Encontra e lista todos os arquivos de imagem no diretório de entrada
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    with tqdm(total=len(files), desc="Processando imagens") as pbar:
        for file in files:
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir,f'mask_{os.path.splitext(file)[0]}.png')
            try:
                #Abre imagem
                img = Image.open(input_path).convert('RGB')          
                #Binariza e salva imagem
                exg_image = ExG(img)
                binary_img = binarize(exg_image)
                binary_img.save(output_path)
            except Exception as e:
                logging.error(f"Erro ao processar {input_path}: {e}")

            pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Constrói dataset para segmentação de vegetação a partir da binarização de imagens.')
    parser.add_argument('--input', required=True, help='Diretório contendo imagens RGB.')
    parser.add_argument('--output', required=True, help='Diretório para salvar máscaras binarizadas.')
    args = parser.parse_args()

    # Configuração do logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    process_images(args.input, args.output)
