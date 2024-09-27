# Reconocimiento de comandos de voz con Pytorch usando Redes Neuronales Recurrentes
El uso de redes recurrentes (RNN) para la modelización de secuencias depende fuertemente del largo de las secuencias de entrada. En este proyecto se investigará cómo diferentes longitudes de secuencia afectan el desempeño de las redes recurrentes en términos de precisión, estabilidad de entrenamiento, y capacidad de generalización.

# Redes Neuronales Recurrentes RNN
The goal of this project is to implement an audio classification system, which:
1. First reads in an audio clip (containing at most one word),
2. Recognizes the class(label) of this audio.

#Librerias
```
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

```


#Paso 2 Creacion de algunos programas utiles

```
#vamos a tener 2 listas con identificaciones, una para validacion y otra para 
carpeta_principal = 'archive-2'
# Obtener nombres de todas las carpetas dentro de la carpeta principal
labels_disponibles = [nombre for nombre in os.listdir(carpeta_principal) if os.path.isdir(os.path.join(carpeta_principal, nombre))]

labels_disponibles.remove('_background_noise_')

columnas = ['Nombre', 'Label', 'Particion', 'Vector_Caracteristicas']

texto_validacion, texto_test = 'archive-2/validation_list.txt', 'archive-2/testing_list.txt'

df = pd.DataFrame(columns=columnas)


#Creamos 2 listas con los nombres de validacion y test
archivos_test = []
archivos_validacion = []

with open(texto_test, 'r') as file:
    for line in file:
        division = line.split('/')
        archivos_test.append(division[-1].strip())

with open(texto_validacion, 'r') as file:
    for line in file:
        division = line.split('/')
        archivos_validacion.append(division[-1].strip())

#Recorremos todas las carpetas y llenamos un DataFrame con su informacion





loc = 0
for carpeta in labels_disponibles:
    ruta = carpeta_principal + '/' + carpeta + '/'
    for archivo in os.listdir(ruta):
        #Leemos el audio

        n_fft = 512
        win_length = 512
        hop_length = 256
        n_mels = 64
        audio, sample_rate = torchaudio.load(ruta+archivo)

        mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
            mel_scale="htk",
        )


        spec = mel_spectrogram(audio)
        spec = librosa.power_to_db(spec)
        spec = spec.flatten()

        if archivo in archivos_test:
            nueva_fila = pd.DataFrame([[archivo, carpeta, 'test', None]], columns=['Nombre', 'Label', 'Particion', 'Vector_Caracteristicas'])
            df = pd.concat([df, nueva_fila], ignore_index=True)
            df.at[loc, 'Vector_Caracteristicas'] = ','.join(map(str, spec))
            loc += 1

        elif archivo in archivos_validacion:
            nueva_fila = pd.DataFrame([[archivo, carpeta, 'validacion', None]], columns=['Nombre', 'Label', 'Particion', 'Vector_Caracteristicas'])
            df = pd.concat([df, nueva_fila], ignore_index=True)
            df.at[loc, 'Vector_Caracteristicas'] = ','.join(map(str, spec))
            loc += 1
        else:
            nueva_fila = pd.DataFrame([[archivo, carpeta, 'train', None]], columns=['Nombre', 'Label', 'Particion', 'Vector_Caracteristicas'])
            df = pd.concat([df, nueva_fila], ignore_index=True)
            df.at[loc, 'Vector_Caracteristicas'] = ','.join(map(str, spec))
            loc += 1



df.to_csv('Vector_caracteristicas.csv', index=False)
```
