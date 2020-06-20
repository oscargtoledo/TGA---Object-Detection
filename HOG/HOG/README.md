# Instrucciones de uso 
Este programa ha sido implementado con Python y CUDA. Por tanto, hace falta una instalación de Python 3.7.7 con los siguientes modulos:
* pycuda
* numpy
* scikit-image
* matplotlib
Si se tiene Python instalado, es tan sencillo como ejecutar los siguientes comandos:
> pip install pycuda

> pip install numpy

> pip install scikit-image

> pip install matplotlib

En su defecto, si no se quiere instalar Python se puede usar el Jupyter Notebook adjuntado, HOGCuda.ipynb. Esto se puede ejecutar localmente con una instalación de Jupyter, o se puede importar en [Google Collaboratory](https://colab.research.google.com), aunque se necesitarán subir los archivos de imagen necesarios.

En cuanto a la ejecución en si, este programa admite 2 parámetros opcionales.
* -file <dirección>, indica que imagen se quiere procesar. Por defecto, es campus.jpg.
* -showImages <cierto/falso>, indica si se muestran las imagenes de cada paso. Por defecto, falso.
El final de la ejecución es un bucle que itera por todos los histogramas generados. Se puede pasar al siguiente histograma cerrando la ventana emergente actual. Para parar la ejecución del programa, se puede hacer Ctrl+C en la consola, cerrar la consola, o parar  desde la instancia del Jupyter Notebook.

