import streamlit as st
import pandas as pd
import joblib
import os

import os
st.write("Directorio actual:", os.getcwd())
st.write("Archivos allí:", os.listdir())

try:
  LogisticRegression_model = joblib.load('LogisticRegression_pipeline.pkl')
  LinearSVC_model = joblib.load('LinearSVC_pipeline.pkl')
  KNeighborsClassifier_model = joblib.load('KNeighborsClassifier_pipeline.pkl')
  DecisionTreeClassifier_model = joblib.load('DecisionTreeClassifier_pipeline.pkl')
  SMV_model = joblib.load('SMV_pipeline.pkl')
  VotingClassifier_model = joblib.load('VotingClassifier_pipeline.pkl')
  RandomForestClassifier_model = joblib.load('RandomForestClassifier_pipeline.pkl')
  XGBoostClassifier_model = joblib.load('XGBoostClassifier_pipeline.pkl')
except FileNotFoundError:
  st.error("El archivo no se encuentra en la ruta especificada.")
  st.stop()

st.title("Aplicación de Predicción de Tipo de Violencia en la ciudad de Medellín y sus corregimientos")
st.write("Sube un archivo Excel para realizar predicciones.")

uploaded_file = st.file_uploader("Carga tu archivo Excel", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Read DataFrame
        df = pd.read_excel(uploaded_file)

        st.subheader("Datos cargados:")
        st.write(df.head())

        # Make predictions using different models
        st.subheader("Predicciones:")

        # Logistic Regression Classifier
        Lr_predictions = LogisticRegression_model.predict(df)
        df['Predicted_LR_Naturaleza'] = Lr_predictions
        st.write("Predicciones (Logistic Regression Classifier): ")
        st.write(df[['edad_', 'sexo_', 'nombre_comuna', 'Predicted_LR_Naturaleza']].head())

        # Linear SVC
        LSVC_predictions = LinearSVC_model.predict(df)
        df['Predicted_LinearSVC_Naturaleza'] = LSVC_predictions
        st.write("Predicciones (Linear SVC):")
        st.write(df[['edad_', 'sexo_', 'nombre_comuna', 'Predicted_LinearSVC_Naturaleza']].head())

        # KNeighbors Classifier
        KNN_predictions = KNeighborsClassifier_model.predict(df)
        df['Predicted_KNN_Naturaleza'] = KNN_predictions
        st.write("Predicciones (KNN):")
        st.write(df[['edad_', 'sexo_', 'nombre_comuna', 'Predicted_KNN_Naturaleza']].head())

        # Decision Tree Classifier
        DT_predictions = DecisionTreeClassifier_model.predict(df)
        df['Predicted_DT_Naturaleza'] = DT_predictions
        st.write("Predicciones (Decision Tree):")
        st.write(df[['edad_', 'sexo_', 'nombre_comuna', 'Predicted_DT_Naturaleza']].head())

        # SMV
        SMV_predictions = SMV_model.predict(df)
        df['Predicted_SVM_Naturaleza'] = SMV_predictions
        st.write("Predicciones (SMV):")
        st.write(df[['edad_', 'sexo_', 'nombre_comuna', 'Predicted_SVM_Naturaleza']].head())

        # Voting Classifier
        Voting_predictions = VotingClassifier_model.predict(df)
        df['Predicted_Voting_Naturaleza'] = Voting_predictions
        st.write("Predicciones (Voting Classifier):")
        st.write(df[['edad_', 'sexo_', 'nombre_comuna', 'Predicted_Voting_Naturaleza']].head())

        # Random Forrest classifier
        RandomForest_predictions = RandomForestClassifier_model.predict(df)
        df['Predicted_RF_Naturaleza'] = randomForest_predictions
        st.write("Predicciones (Random Forrest):")
        st.write(df[['edad_', 'sexo_', 'nombre_comuna', 'Predicted_RF_Naturaleza']].head())

        # XGBoost Classifier
        XGBoost_predictions = XGBoostClassifier_model.predict(df)
        df['Predicted_XGB_Naturaleza'] = XGBoost_predictions
        st.write("Predicciones (XGBoost):")
        st.write(df[['edad_', 'sexo_', 'nombre_comuna', 'Predicted_XGB_Naturaleza']].head())

        st.subheader("Resultados Completos:")
        # Check if the columns exist before trying to display them
        display_cols = ['edad_', 'sexo_', 'nombre_comuna', 'Predicted_LR_Naturaleza', 'Predicted_LinearSVC_Naturaleza', 'Predicted_KNN_Naturaleza', 'Predicted_DT_Naturaleza', 'Predicted_SVM_Naturaleza',
                       'Predicted_Voting_Naturaleza', 'Predicted_RF_Naturaleza']

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
