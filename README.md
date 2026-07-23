# Segmentação de Vegetação com DeepLabV3+

Este projeto realiza a detecção (segmentação) de vegetação em imagens.
Ele é etruturado em 4 etapas:

1. Quebra de Imagem em Blocos;
2. Geração de Dataset;
3. Implementação e Treinamento de Rede Neural;
4. Inferência do Modelo.

### Requisitos

Para desenvolvimento e execução do projeto, foram utilizados Python 3.9.13 e Tensorflow 2.17. Os códigos foram executadas em um sistema operacional Ubuntu 22.04 e ambiente virtual Conda. Além disso, foram utilizados pacotes auxiliares, que podem ser instalados a partir da execução de `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Arquivos auxiliares

No link do [Google Drive](https://drive.google.com/drive/folders/1NwCtDuBmdh11bZdIbU53uIX4_JdfwY-m?usp=sharing) é possível acessar tanto os arquivos utilizados/gerados neste projeto. Eles são:

- `data/Orthomosaico_roi.tif`: arquivo para quebra de imagens em blocos;
- `dataset/`: agrega imagens geradas na quebra da imagem em blocos, também utilizada para binarização e, posteriormente, treinamento da rede;
- `masks/`: máscaras construídas na fase de geração de dataset e utilizadas para treinamento da rede;
- `models/final.h5`: modelo gerado na frase de treinamento da rede.

Contudo, para execução do projeto, é necessário somente o arquivo `Orthomosaico_roi.tif`. Demais arquivos podem ser gerados a partir das etapas listadas a seguir.

## Quebra de imagem em blocos

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

Os dados extraídos indicam que não há necessidade de normalização e processamentos adicionais nas imagens, dado que os valores dos pixels estão entre 0 e 255 e não há dados inválidos. Pode-se observar que há 4 bandas na imagem, porém serão utilizados somente as 3 primeiras, que representam os canais RGB.
### Utilização do script para subdivisão da imagem

Para subdividir a imagem em sub-imagens, basta executar o seguinte comando:

```bash
python divide_orthomosaic.py --input </path/to/orthomosaic.tif> --output </path/to/output/dir/>
```

onde `--input` deve indicar o caminho para a imagem e `--output` o diretório onde as imagens serão salvas.

### Exemplo de imagens subdivididas

A imagem original foi sub-dividia em imagens de tamanho 256x256, mantendo o canal de transparência em regiões de borda. Abaixo estão ilustrados alguns exemplos de imagens resultantes da divisão:

![](doc/image_3.png)   ![](doc/image_39.png)  ![](doc/image_54.png)

![](doc/image_194.png)   ![](doc/image_195.png)  ![](doc/image_198.png) 

![](doc/image_200.png)   ![](doc/image_201.png)  ![](doc/image_202.png) 


## Geração de Dataset

Para gerar o dataset com máscaras de segmentação são utilizadas as sub-imagens geradas na etapa anterior. O método adotado para gerar as máscaras foi o ExG (Excess Green Index), que analisa os canais Vermelho (R), Verde (G) e Azul (B) e torna evidente o canal verde. Isto se dá através da formula `2 * g − r − b`. Logo após esta operação, é utilizado um limiar (definido como 128) visando binarizar a máscara resultante. Para gerar o dataset, basta executar o seguinte comando:

```bash
python binarize_images.py --input </path/to/images/dir> --output </path/to/segmented/dir/>
```
onde `--input` deve indicar o caminho para o diretório das sub-imagens e `--output` o diretório onde as máscaras serão salvas.

### Exemplo de imagens e respectivas máscaras

Imagem RGB            |  Máscara (ground truth)
:-------------------------:|:-------------------------:
![](doc/image_3.png)  |  ![](doc/mask_image_3.png)

Imagem RGB            |  Máscara (ground truth)
:-------------------------:|:-------------------------:
![](doc/image_54.png)  |  ![](doc/mask_image_54.png)

Imagem RGB            |  Máscara (ground truth)
:-------------------------:|:-------------------------:
![](doc/image_201.png)  |  ![](doc/mask_image_201.png)


## Implementação e Treinamento de Rede Neural

Para treinar um modelo de segmentação semântica para vegetação, foi selecionado um modelo pré-treinado, especificamente o modelo DeepLabV3+ com backbone da ResNet50. A utilização do modelo pré-treinado permite uma boa generalização do modelo logo nas primeiras épocas de treinamento, utilizando poucos exemplos. Para esta modelagem, foram utilizadas as bibliotecas Tensorflow e Keras. Foi utilizado o otimizador Adam e as métricas Intersection over Union (IoU) e acurácia para avaliar o treinamento do modelo. 

### Parâmetros de treinamento

- Tamanho das imagens: 256x256
- Batch Size: 32
- Épocas: 200
- Imagens de treinamento: 408
- Imagens de validação: 72
- Otimizador: Adam
- Rede pré-treinada: DeepLabV3+ com ResNet50 como backbone
- Métricas de avaliação: BinaryIoU e Accuracy

Algumas destas configurações, tais como batch size, épocas, taxa de divisão (validação/treinamento), dentre outras, podem ser configuradas em `config/config.py`.

### Estrutura do código

- `net/resnet50.py`: implementa classe `SegmentationResnet50`, responsável por agrupar diferentes funções para carregar modelo pré-treinado, carregar dataset, salvar modelo, gerar inferências a partir de imagens, etc.
- `config/config.py`: agrupa parâmetros de treinamento e outras configurações;
- `train_model.py`: gerencia parâmetros de execução, instancia classe `SegmentationResNet50`, treina e salva o modelo.

### Treinamento

Para realizar o treinamento do modelo e salvá-lo, execute o seguinte comando:

```bash
python train_model.py --rgb </path/to/images/dir> --groundtruth </path/to/segmented/dir/> --modelpath </path/to/model.h5>
```

onde `--rgb` indica o diretório com dataset RGB, `--groundtruth` as respectivas máscaras do dataset, e `--modelpath` diretório/nome do modelo a ser salvo ao final do treinamento. 

### Resultados de treinamento

Durante o treinamento do modelo, algumas métricas foram calculadas, tanto para dados de treino quanto de validação. As métricas utilizadas foram Acurácia e BinaryIoU. Esta última visa avaliar o quanto a inferência é similar ao ground truth.



Acurácia            |  BinaryIoU
:-------------------------:|:-------------------------:
![](doc/acuracia.png)  |  ![](doc/iou.png)


Note que, em dado momento, por volta da época 126, as métricas se estabilizam, com acurácia chegando a aproximadamente 94,43% e BinaryIoU em 87,82%, ambas nos dados de validação.

## Inferência do Modelo

Após o treinamento do modelo, é possível carregar e utilizar os pesos treinados para segmentação de imagens com vegetação. Para isto, é importante respeitar a mesma escala utilizada para treinamento. Imagens com resoluções baixas podem não ter resultados precisos.

Apesar das imagens utilizadas para treinamento da rede possuírem dimensões de 256x256 pixels, é possível inferir imagens de dimensões superiores. Para isto, o algoritmo de inferência faz a leitura da imagem em janelas menores e realiza a predição a partir destas sub-imagens. Antes de executar a operação em questão, uma matriz de mesma dimensão da imagem de entrada é inicializa com valores zerados. Conforme o algoritmo realiza o janelamento e a predição, a matriz auxiliar é preenchida com as máscaras resultantes. Ao final, essa matriz é salva como uma imagem binária. 

A binarização da imagem é necessária dado que a rede gera valores entre 0 e 1, indicando a "confiança" de que dados pixels possam pertencer a classe vegetada (1) e não vegetada (0). Desta forma, foi definido o valor de 0,9 como limiar da binarização. Este valor foi selecionado pois torna a segmentação da vegetação um pouco mais discreta. Entretanto, é possível configurar este limiar em `config/config.py`.

### Execução da inferência

Para executar a inferência e gerar uma máscara segmentada para vegetações, basta executar:

```bash
   python model_inference.py --rgb </path/to/image.png> --modelpath </path/to/model.h5> --output </path/to/segmented/image.png>
```

onde `--rgb` indica a imagem a ser predita, `--modelpath` o caminho do modelo previamente treinado, e `--output` diretório/nome que a máscara gerada deverá ter ao ser salva. 

### Resultado com imagem ortomoisaica de canavial

A partir de uma imagem obtida em [lapix.ufsc.br/wp-content/uploads/2019/05/sugarcane2.png](https://lapix.ufsc.br/wp-content/uploads/2019/05/sugarcane2.png), foi possível testar o modelo treinado. Assim como descrito anteriormente, a imagem é sub-dividida em sub-imagens e a inferência é realizada por janelamento. O modelo gera valores entre 0 e 1 para cada pixel da imagem, sendo assim, é necessário binarizar o resultado para gerar uma máscara binária. O valor do limiar escolhido foi de 0.9, que pode ser configurado em `config/config.py`.

Abaixo, é possível visualizar a imagem de teste e respectiva máscara obtida ao ser analisada pela rede treinada. O resultado obtido é bastante robusto, ainda mais considerando as diferenças entre imagens utilizadas para treinamento e a imagem de teste.

Imagem de Teste            |  Segmentação obtida
:-------------------------:|:-------------------------:
![](doc/sugarcane2.png)  |  ![](doc/result.png)



## Discussões Sobre o Resultado e Pontos de Melhoria

Os resultados obtidos durante o treinamento e na fase de inferência indicam uma ótima performance da rede de segmentação. A utilização de uma arquitetura de rede com pesos pré-treinados possibilitou treinar a rede por poucas épocas de treino e poucos exemplos de dados.

Como apresentado anteriormente, a rede teve uma acurácia de cerca de 94,43% e BinaryIoU em 87,82% em seu ápice, considerando o conjunto de validação durante o treino. Na fase de inferência, foi utilizada uma imagem obtida na literatura de uma área com canavial. Como esta imagem não possui máscara de segmentação associada, não foi possível calcular métricas de eficiência na inferência de tal imagem. Contudo, análises visuais indicam que a rede conseguiu generalizar muito bem para um caso de teste com configurações diferentes de imagem, captura, etc.


 Apesar dos ótimos resultados, há alguns pontos que não foram explorados, tanto para modelagem e construção da rede quanto para sua configuração. Além da rede, há diversos fatores que podem ser considerados nas etapas de tratamento dos dados e construção do dataset. Alguns desses pontos são:  

- **Avaliar outros métodos de binarização dos dados:** para este projeto foi utilizado o método ExG para auxiliar na construção do dataset de treinamento da rede. Há diversos outros métodos que podem ser testados, bem como, a literatura aponta outras formas de implementar o ExG. Além disso, foi utilizado um limiar para binarizar as máscara de segmentação final. Esse valor gera alterações na máscara gerada e pode afetar o treinamento da rede.

- **Avaliar outras arquiteturas de rede:** foi utilizada a rede DeepLabV3+ com backbone da ResNet50 e pesos pré-treinados, o que garantiu uma boa performance da rede, mesmo utilizando poucos exemplos para treinamento. Entretanto, há diversas arquiteturas de rede na literatura que podem ser testadas e que podem trazer benefícios à segmentação.

- **Verificar outros parâmetros de treinamento:** há diversas configurações e hiper-parâmetros que podem ser testados e que podem alterar o comportamento do treinamento da rede.

- **Avaliar performance do modelo utilizando outros otimizadores:** para o projeto foi utilizado o otimizador ADAM, que permitiu simplificar a configuração do código e trouxe bons resultados. Contudo, há diversos outros otimizadores que poderiam ser testados, tais como, SGD, RMSProp, AdaGrad, Adadelta, dentre outros.

- **Aplicar Data Augmentation no dataset:** o dataset gerado possui 480 imagens, tendo parcela reservada para validação e o restante para treinamento do modelo. Tal quantidade é insuficiente para treinamento da maioria dos modelos de redes neurais, principalmente para problemas complexos. Como foi utilizado uma rede pré-treinada, a não generalização não foi um problema para as validações realizadas. Contudo, a utilização de Data Augmentation pode trazer benefícios na generalização do conhecimento para uma variedade maior de tipos de vegetação, mudança de luminosidade, sombras, etc.


- **Utilizar outras métricas de validação:** para o problema de segmentação, é ideal utilizar métricas que visam comparar as máscaras inferidas contra as máscaras esperadas. Para o projeto, foi utilizada a BinaryIoU para auxiliar nessa comparação. Também foi utilizada a acurácia que simplifica o processo de avaliação do aprendizado. Além destas métricas, poderiam ser utilizas a Dice Loss, Dice Coefficient, Jaccard Index, acurácia por Pixel, dentre outras. 

- **Recuperar modelo com melhor performance no conjunto de validação durante o treino:** atualmente são utilizados os pesos gerados na última época de treinamento. Em casos de treinamentos longos, estes pesos podem estar sobreajuste aos dados de treinamento, além de outros problemas. O ideal seria considerar os pesos que melhor performam no conjunto de validação durante o treinamento.

- **Verificar o valor ideal para binarizar inferências:** após o treinamento, a saída da rede gera uma máscara onde cada valor de pixel está entre 0 e 1, onde valores próximos a 1 indicam uma confiança maior de que aquele pixel se refere à classe 1. Para binarizar o resultado obtido na etapa de Inferência do Modelo, foi utilizado um limiar de 0,9, o que torna mais discreta a segmentação para a região vegetada, contudo, pode restringir a segmentação de regiões com pouca densidade de vegetação.
