import streamlit as st
import pandas as pd
import joblib
import os

try:
  logisticRegression_model = joblib.load('LogisticRegression_pipeline.pkl')
  linearSVC_model = joblib.load('LinearSVC_pipeline.pkl')
  KNeighborsClassifier_model = joblib.load('KNeighborsClassifier_pipeline.pkl')
  DecisionTreeClassifier_model = joblib.load('DecisionTreeClassifier_pipeline.pkl')
  svm_model = joblib.load('SMV_pipeline.pkl')
  votingClassifier_model = joblib.load('VotingClassifier_pipeline.pkl')
  randomForestClassifier_model = joblib.load('RandomForestClassifier_pipeline.pkl')
  XGBoostClassifier_model = joblib.load('XGBoostClassifier_pipeline.pkl')
except FileNotFoundError:
  st.error("El archivo no se encuentra en la ruta especificada.")
  st.stop()

st.title("Aplicación de Predicción de Tipo de Violencia en la ciudad de Medellín y sus corregimientos")
st.write("Sube un archivo Excel para realizar predicciones.")

uploaded_file = st.file_uploader("Carga tu archivo Excel", type=["xlsx"])

if uploaded_file is not None:
  df = pd.read_excel(uploaded_file)
  st.write("Vista previa de los datos cargados:")
  st.dataframe(df.head())
else:
  st.write("Por favor, sube un archivo Excel para continuar.")

if 'df' in locals():
  predictions = {}
  models = {
      "Logistic Regression": logisticRegression_model,
      "Linear SVC": linearSVC_model,
      "KNeighbors Classifier": KNeighborsClassifier_model,
      "Decision Tree Classifier": DecisionTreeClassifier_model,
      "SVM": svm_model,
      "Voting Classifier": votingClassifier_model,
      "Random Forest Classifier": randomForestClassifier_model,
      "XGBoost Classifier": XGBoostClassifier_model
  }
  for model_name, model in models.items():
    try:
      predictions[model_name] = model.predict(df)
    except Exception as e:
      st.warning(f"Could not make predictions with {model_name}: {e}")

if 'predictions' in locals() and predictions:
  predictions_df = pd.DataFrame(predictions)
  st.write("Tabla comparativa de predicciones:")
  st.dataframe(predictions_df)
else:
  st.write("No se pudieron generar predicciones. Por favor, asegúrate de que el archivo cargado sea correcto y los modelos estén disponibles.")

if 'df' in locals():
  st.write("Selecciona los modelos para la predicción:")
  models = {
      "Logistic Regression": logisticRegression_model,
      "Linear SVC": linearSVC_model,
      "KNeighbors Classifier": KNeighborsClassifier_model,
      "Decision Tree Classifier": DecisionTreeClassifier_model,
      "SVM": svm_model,
      "Voting Classifier": votingClassifier_model,
      "Random Forest Classifier": randomForestClassifier_model,
      "XGBoost Classifier": XGBoostClassifier_model
  }
  selected_models = st.multiselect(
      "Modelos disponibles",
      list(models.keys()),
      list(models.keys()) # Default to all models selected
  )

  predictions = {}
  if selected_models:
    for model_name in selected_models:
      if model_name in models:
        model = models[model_name]
        try:
          predictions[model_name] = model.predict(df)
        except Exception as e:
          st.warning(f"Could not make predictions with {model_name}: {e}")
      else:
          st.warning(f"Model '{model_name}' not found.")

    if predictions:
      predictions_df = pd.DataFrame(predictions)
      st.write("Tabla comparativa de predicciones:")
      st.dataframe(predictions_df)
    else:
      st.write("No se pudieron generar predicciones para los modelos seleccionados.")
  else:
    st.write("Por favor, selecciona al menos un modelo para generar predicciones.")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if 'df' in locals() and 'predictions' in locals() and predictions:
  st.write("## Evaluación de Modelos")
  # Prompt user for the true label column
  label_column = st.text_input("Ingresa el nombre de la columna con las etiquetas verdaderas:", "")

  if label_column and label_column in df.columns:
    true_labels = df[label_column]
    metrics = {}
    for model_name, preds in predictions.items():
      try:
        # Ensure true_labels and predictions are of compatible types and shapes
        if len(true_labels) == len(preds):
          metrics[model_name] = {
              "Accuracy": accuracy_score(true_labels, preds),
              "Precision": precision_score(true_labels, preds, average='weighted', zero_division=0),
              "Recall": recall_score(true_labels, preds, average='weighted', zero_division=0),
              "F1 Score": f1_score(true_labels, preds, average='weighted', zero_division=0)
          }
        else:
          st.warning(f"Skipping metrics for {model_name}: Mismatch in length between true labels and predictions.")
      except Exception as e:
        st.warning(f"Could not calculate metrics for {model_name}: {e}")

    if metrics:
      metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
      st.write("Métricas de rendimiento de los modelos:")
      st.dataframe(metrics_df)
    else:
      st.write("No se pudieron calcular métricas de rendimiento para los modelos seleccionados.")
  elif label_column:
    st.warning(f"La columna '{label_column}' no se encuentra en el archivo cargado.")
  else:
    st.write("Ingresa el nombre de la columna con las etiquetas verdaderas para ver las métricas de rendimiento.")
