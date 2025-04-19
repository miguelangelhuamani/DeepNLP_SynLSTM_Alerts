# Práctica Final de Deep Learning y NLP

Este proyecto forma parte de la asignatura de Deep Learning y Procesamiento de Lenguaje Natural (NLP). El objetivo es desarrollar un sistema que combine el Reconocimiento de Entidades Nombradas (NER) y el Análisis de Sentimiento (SA) para generar alertas automáticas a partir de artículos de noticias y publicaciones en redes sociales.

## Descripción del Proyecto

El sistema desarrollado procesa textos para identificar entidades como personas, organizaciones, valores monetarios, ubicaciones, etc., utilizando una red LSTM personalizada. Además, clasifica el sentimiento del texto (positivo, neutral o negativo) mediante otra red neuronal. Finalmente, combina las salidas de NER y SA para generar alertas contextuales relevantes.

## Estructura del Repositorio

- `data/`: Contiene los conjuntos de datos utilizados en el proyecto.
- `src/`: Incluye el código fuente del proyecto.
- `conll2003_data.pkl`: Archivo serializado con los datos preprocesados del conjunto CoNLL-2003.


## Instalación

1. **Clonar el repositorio**:

   ```bash
   git clone https://github.com/ndelval/PracticaFinalDeepNLP.git
   ```

2. **Instalar las dependencias**:

   Se recomienda crear un entorno virtual para gestionar las dependencias:

   ```bash
   cd PracticaFinalDeepNLP
   python3 -m venv env
   source env/bin/activate  # En Windows: 'env\Scripts\activate'
   ```

   Luego:

   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

## Uso

### Entrenamiento de Modelos

Para entrenar los modelos de NER y SA:

```bash
python src/train.py
```

Los pesos entrenados se guardarán en `models/`.

### Evaluación

Para evaluar los modelos:

```bash
python src/evaluate.py
```

### Generación de Alertas

Para generar alertas a partir de nuevos textos:

```bash
python src/generate_alerts.py --input_path path/to/input.txt --output_path path/to/alerts.txt
```

## Conjunto de Datos

Se utiliza el conjunto de datos CoNLL-2003 para NER. El archivo `conll2003_data.pkl` debe estar en el directorio principal. 
## Contribuciones

1. Haz un fork del repositorio
2. Crea una rama nueva: `git checkout -b feature/nueva-funcionalidad`
3. Realiza los cambios y haz commit: `git commit -am 'Nueva funcionalidad'`
4. Sube los cambios: `git push origin feature/nueva-funcionalidad`
5. Abre un Pull Request


## Contacto

Para consultas o sugerencias, contactar con el equipo del proyecto o el profesor responsable de la asignatura.

