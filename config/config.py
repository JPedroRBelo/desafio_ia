# VARIÁVEIS DE CONFIGURAÇÂO

#Tamanho dos subtiles utilizados em divide_orthomosaic
SUBTILES_SIZE = (256,256)

#Limiar para binarizar máscara de segmentação para ExG
BINARIZE_THRESHOLD = 75

#Limiar para binarização de inferências
MASK_THRESHOLD = 0.9

#Valor do pixel da classe segunda classe binária (vegetação)
BINARY_MAX = 1

#Treinamento 
IMG_WIDTH = 256
IMG_HEIGHT = 256
BATCH_SIZE = 32
EPOCHS = 200
TEST_SPLIT = 0.0
VAL_SPLIT = 0.15