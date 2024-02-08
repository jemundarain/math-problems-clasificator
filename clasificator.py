import pandas as pd
import numpy as np
import nltk
import joblib

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from collections import defaultdict
from nltk.corpus import wordnet as wn

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Cargamos el conjunto de datos desde un archivo CSV
Corpus = pd.read_csv(r"problems-training.csv", encoding='utf-8')

# Paso 1: Preprocesamiento de texto
# 1.1: Eliminar filas en blanco, si las hay
Corpus['Statement'].dropna(inplace=True)

# 1.2: Convertir todo el texto a minúsculas
Corpus['Statement'] = [entry.lower() for entry in Corpus['Statement']]

# 1.3: Segmentación
Corpus['Statement'] = [word_tokenize(entry) for entry in Corpus['Statement']]

# 1.4: Eliminar palabras vacías, no numéricas y realizar lematización.
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

Corpus['text_final'] = ""
for index, entry in enumerate(Corpus['Statement']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
        Final_words.append(word_Final)
    Corpus.loc[index, 'text_final'] = ' '.join(Final_words)

# Paso 2: Dividir el conjunto de datos en entrenamiento y prueba
X = Corpus['text_final']  # Los datos de entrada, los textos preprocesados
y = Corpus['Category']  # Las etiquetas de clase

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Paso 3: Crear un vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Paso 4: Entrenar un modelo SVM
model = SVC(kernel='linear')
model.fit(X_train_tfidf, y_train)

# Paso 5: Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test_tfidf)

# Paso 6: Guardar el modelo entrenado utilizando joblib
joblib.dump(model, 'modelo_svm.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

# Paso 7: Evaluar el modelo
# 7.1: Calcular la matriz de confusión
confusion = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(confusion)

# 7.2: Generar un informe de clasificación
target_names = ['álgebra', 'geometría', 'combinatoria', 'teoría de números']
report = classification_report(y_test, y_pred, target_names=target_names)
print("\nInforme de Clasificación:")
print(report)