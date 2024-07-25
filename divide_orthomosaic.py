import argparse
import os
import logging
from PIL import Image
from tqdm import tqdm
from config.config import SUBTILES_SIZE

def subdivide_tiff_image(input_path: str, output_dir: str, tile_size: tuple[int, int]) -> None:
    """
    Subdivide uma imagem TIFF em sub-imagens menores e as salva como arquivos PNG.

    Args:
        input_path (str): Caminho para arquivo TIFF de entrada.
        output_dir (str): Caminho para diretório onde as sub-imagens serão salvas.
        tile_size (tuple): Tamanho de cada sub-imagem (largura, altura).

    Returns:
        None
    """

    #Abre imagem
    img = Image.open(input_path)
    img_width, img_height = img.size
    tile_width, tile_height = tile_size
    
    #Auxilia na denominação de novos subtiles
    count = 0

    #Calcula a quantidade total de subtiles a serem gerados
    total_tiles = (img_width // tile_width + (1 if img_width % tile_width != 0 else 0)) * (img_height // tile_height + (1 if img_height % tile_height != 0 else 0))

    with tqdm(total=total_tiles, desc="Subdividindo imagem") as pbar:
        for i in range(0, img_width, tile_width):
            for j in range(0, img_height, tile_height):
                #Calcula box a ser recortado da imagem de origem
                box = (i, j, i + tile_width, j + tile_height)
                tile = img.crop(box)

                #Salva subtile em disco
                tile_filename = os.path.join(output_dir, f"image_{count}.png")
                tile.save(tile_filename)
                count += 1
                pbar.update(1)

    logging.info(f"Subimagens salvas em: {output_dir}")

def main(input_path: str, output_dir: str) -> None:
    """
    Código principal para gerenciar leitura, subdivisão e escrita de imagens.

    Args:
        input_path (str): Caminho para arquivo ortomosaico. Exemplo: /path/to/Orthomosaico_roi.tif.
        output_dir (str): Caminho para diretório onde os resultados serão salvos. Exemplo: /path/to/output/dir/.

    Returns:
        None

    Raises:
        FileNotFoundError: Se o arquivo de entrada não existir.
        OSError: Se o diretório de saída não puder ser criado.
    """

    # Verifica se arquivo de entrada existe
    if not os.path.isfile(input_path):
        logging.error(f"Erro: O arquivo de entrada {input_path} não existe.")
        return
    
    # Cria diretório de saída, caso não exista
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    subdivide_tiff_image(input_path,output_dir,tile_size=SUBTILES_SIZE)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='quebrar uma imagem TIFF em partes menores.')
    parser.add_argument('--input', required=True, help='Caminho para o arquivo ortomosaico. Por exemplo: /path/to/Orthomosaico_roi.tif')
    parser.add_argument('--output', required=True, help='Caminho para o diretório de saída. Por exemplo: /path/to/output/dir/')    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    main(args.input, args.output)