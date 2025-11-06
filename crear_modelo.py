import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

def entrenar_modelo():
    """
    Función para entrenar y guardar el modelo de Regresión Logística
    """
    print("🔄 Cargando y preparando datos...")
    
    # CARGA tu dataset real:
    try:
        df = pd.read_csv('heart_cleaned_final.csv')
        print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    except FileNotFoundError:
        print("❌ Error: No se encontró 'heart_cleaned_final.csv'")
        print("📋 Creando datos de ejemplo temporalmente...")
        # Datos de ejemplo temporal
        np.random.seed(42)
        n_samples = 297
        data = {
            'age': np.random.randint(29, 77, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'cp': np.random.randint(0, 4, n_samples),
            'trestbps': np.random.randint(94, 200, n_samples),
            'chol': np.random.randint(126, 564, n_samples),
            'fbs': np.random.randint(0, 2, n_samples),
            'restecg': np.random.randint(0, 3, n_samples),
            'thalach': np.random.randint(71, 202, n_samples),
            'exang': np.random.randint(0, 2, n_samples),
            'oldpeak': np.round(np.random.uniform(0, 6.2, n_samples), 1),
            'slope': np.random.randint(0, 3, n_samples),
            'ca': np.random.randint(0, 4, n_samples),
            'thal': np.random.randint(1, 4, n_samples),
            'diagnostico': np.random.randint(0, 2, n_samples)
        }
        df = pd.DataFrame(data)
    
    # Separar características y objetivo
    X = df.drop('diagnostico', axis=1)
    y = df['diagnostico']
    
    print("🎯 Entrenando modelo de Regresión Logística...")
    
    # Crear pipeline con StandardScaler y LogisticRegression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Entrenar el modelo
    pipeline.fit(X, y)
    
    # Guardar el modelo
    with open('pipeline_reglog_13f.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    
    print("✅ Modelo guardado como 'pipeline_reglog_13f.pkl'")
    
    # Mostrar métricas básicas
    accuracy = pipeline.score(X, y)
    print(f"📊 Accuracy en entrenamiento: {accuracy:.3f}")
    
    # Mostrar coeficientes
    if hasattr(pipeline.named_steps['logreg'], 'coef_'):
        print("🔍 Coeficientes del modelo:")
        for i, col in enumerate(X.columns):
            coef = pipeline.named_steps['logreg'].coef_[0][i]
            print(f"   {col}: {coef:.3f}")

if __name__ == "__main__":
    entrenar_modelo()