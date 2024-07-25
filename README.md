# Desafio IA

Este repositório é referente a um desafio para detecção (segmentação) de vegetação.
Ele é etruturado em 4 etapas:

- preparação dos dados;
- geração de dataset;
- treinamento do modelo;
- validação (inferência utilizando modelo treinado).

## Preparação dos Dados

### Informações sobre dados utilizados

Após análise da imagem `Orthomosaico_roi.tif` no QGIS, foi possível observar as seguintes informações sobre a imagem:

- Largura: 7572
- Altura: 3849
- Tipo de dado: Byte - Inteiro de 8 bits sem sinal
- Metadados do driver GDAL: GeoTIFF
- Nº de Bandas: 4
- Escala: 1
- Banda 1:
    - STATISTICS_MAXIMUM=255
    - STATISTICS_MEAN=147.59185051023
    - STATISTICS_MINIMUM=0
    - STATISTICS_STDDEV=37.511229713501
    - STATISTICS_VALID_PERCENT=100
- Banda 2: 
    - STATISTICS_MAXIMUM=251
    - STATISTICS_MEAN=146.4752618251
    - STATISTICS_MINIMUM=0
    - STATISTICS_STDDEV=36.13321651472
    - STATISTICS_VALID_PERCENT=100
- Banda 3: 
    - STATISTICS_APPROXIMATE=YES
    - STATISTICS_MAXIMUM=254
    - STATISTICS_MEAN=142.3536357843
    - STATISTICS_MINIMUM=0
    - STATISTICS_STDDEV=35.446070604004
    - STATISTICS_VALID_PERCENT=100
- Banda 4: 
    - STATISTICS_MAXIMUM=255
    - STATISTICS_MEAN=250.93260257088
    - STATISTICS_MINIMUM=0
    - STATISTICS_STDDEV=31.94749790798
    - STATISTICS_VALID_PERCENT=100

### Utilização do script para subdivisão da imagem

Para subdividir a imagem em subimagens, basta executar o seguinte comando:

```bash
    python divide_orthomosaic.py --input </path/to/orthomosaic.tif> --output </path/to/output/dir/>
```

onde `</path/to/orthomosaic.tif>` deve indicar o caminho para a imagem e `</path/to/output/dir/>` o diretório onde as imagens serão salvas.

## Geração de Dataset

Para gerar o dataset com máscaras de segmentação são utilizadas as subimagens geradas na etapa anterior. O método adotado para gerar as máscaras foi o ExG (Excess Green Index), que analisa os canais Vermelho (R), Verde (G) e Azul (B) e torna evidente o canal verde. Isto se dá através da formula `2 * g − r − b`. Logo após esta operação, é utilizado um limiar (definido como 128) visando binarizar a máscara resultante. Para gerar o dataset, basta executar o seguinte comando:

```bash
   python binarize_images.py --input </path/to/images/dir> --output </path/to/segmented/dir/>
```
onde `</path/to/images/dir>` deve indicar o caminho para o diretório das subimagens e `</path/to/segmented/dir/>` o diretório onde as máscaras serão salvas.

## Treinamento do Modelo

## Validação

## Discussão sobre o Desafio
