# Демонстрационный скрипт для тестирования приложения
# Этот скрипт проверяет основные функции без запуска Streamlit

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def test_data_loading():
    """Тестирование загрузки данных"""
    print("🔄 Тестирование загрузки данных...")

    try:
        # Пробуем загрузить локальные данные
        data = pd.read_csv('data/predictive_maintenance.csv')
        print(f"✅ Локальные данные загружены: {data.shape}")
        return data
    except FileNotFoundError:
        print("❌ Локальные данные не найдены")

        try:
            # Пробуем загрузить из UCI
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=601)
            data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
            print(f"✅ Данные UCI загружены: {data.shape}")
            return data
        except ImportError:
            print("❌ ucimlrepo не установлен")
            print("💡 Установите: pip install ucimlrepo")
            return None
        except Exception as e:
            print(f"❌ Ошибка загрузки UCI: {e}")
            return None

def test_preprocessing(data):
    """Тестирование предобработки данных"""
    print("\n🔄 Тестирование предобработки...")

    if data is None:
        print("❌ Нет данных для обработки")
        return None, None

    # Создаем копию данных
    processed_data = data.copy()

    # Удаляем ненужные столбцы
    columns_to_drop = ['UDI', 'Product ID']
    if 'Machine failure' in processed_data.columns:
        failure_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        columns_to_drop.extend([col for col in failure_columns if col in processed_data.columns])
        target_column = 'Machine failure'
    else:
        target_column = 'Target'

    columns_to_drop = [col for col in columns_to_drop if col in processed_data.columns]
    if columns_to_drop:
        processed_data = processed_data.drop(columns=columns_to_drop)
        print(f"🗑️ Удалены столбцы: {columns_to_drop}")

    # Кодирование категориальной переменной
    if 'Type' in processed_data.columns:
        le = LabelEncoder()
        processed_data['Type'] = le.fit_transform(processed_data['Type'])
        print("🔤 Переменная 'Type' закодирована")

    # Переименуем целевую переменную
    if target_column in processed_data.columns and target_column != 'Target':
        processed_data = processed_data.rename(columns={target_column: 'Target'})

    # Разделение на признаки и целевую переменную
    if 'Target' in processed_data.columns:
        X = processed_data.drop(columns=['Target'])
        y = processed_data['Target']

        # Масштабирование
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        print(f"✅ Предобработка завершена: {X_scaled_df.shape} признаков")
        print(f"📊 Распределение классов: {y.value_counts().to_dict()}")

        return X_scaled_df, y
    else:
        print("❌ Целевая переменная не найдена")
        return None, None

def test_models(X, y):
    """Тестирование обучения моделей"""
    print("\n🔄 Тестирование моделей...")

    if X is None or y is None:
        print("❌ Нет данных для обучения")
        return None

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📈 Обучающая выборка: {len(X_train)}")
    print(f"📉 Тестовая выборка: {len(X_test)}")

    # Модели для тестирования
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42)  # Меньше деревьев для быстроты
    }

    results = {}

    for name, model in models.items():
        print(f"🤖 Обучение {name}...")

        try:
            # Обучение
            model.fit(X_train, y_train)

            # Предсказания
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Метрики
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            conf_matrix = confusion_matrix(y_test, y_pred)

            results[name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'confusion_matrix': conf_matrix
            }

            print(f"  ✅ Accuracy: {accuracy:.3f}")
            print(f"  ✅ ROC-AUC: {roc_auc:.3f}")

        except Exception as e:
            print(f"  ❌ Ошибка: {e}")

    return results

def test_prediction(X, y):
    """Тестирование предсказания на новых данных"""
    print("\n🔄 Тестирование предсказания...")

    if X is None or y is None:
        print("❌ Нет данных для тестирования")
        return

    # Быстрое обучение модели
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Создаем пример новых данных
    new_data = pd.DataFrame({
        col: [X[col].mean()] for col in X.columns
    })

    # Делаем предсказание
    prediction = model.predict(new_data)[0]
    prediction_proba = model.predict_proba(new_data)[0]

    print(f"🔮 Пример предсказания:")
    print(f"  Предсказание: {'ОТКАЗ' if prediction == 1 else 'НОРМА'}")
    print(f"  Вероятность отказа: {prediction_proba[1]:.3f}")
    print(f"  Вероятность нормы: {prediction_proba[0]:.3f}")

def main():
    """Основная функция тестирования"""
    print("🚀 Запуск демонстрационного тестирования")
    print("=" * 50)

    # Тестирование загрузки данных
    data = test_data_loading()

    # Тестирование предобработки
    X, y = test_preprocessing(data)

    # Тестирование моделей
    results = test_models(X, y)

    # Тестирование предсказания
    test_prediction(X, y)

    print("\n" + "=" * 50)
    print("✅ Демонстрационное тестирование завершено!")

    if results:
        print("\n📊 Итоговые результаты:")
        for model_name, metrics in results.items():
            print(f"  {model_name}:")
            print(f"    Accuracy: {metrics['accuracy']:.3f}")
            print(f"    ROC-AUC: {metrics['roc_auc']:.3f}")

if __name__ == "__main__":
    main()
