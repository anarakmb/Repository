import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

def analysis_and_model_page():
    st.title("🔧 Анализ данных и модель предиктивного обслуживания")
    st.markdown("---")

    # Sidebar для настроек
    st.sidebar.header("⚙️ Настройки модели")

    # Вариант загрузки данных
    data_source = st.sidebar.radio(
        "Выберите источник данных:",
        ["Загрузить CSV файл", "Использовать UCI Repository"]
    )

    data = None

    if data_source == "Загрузить CSV файл":
        uploaded_file = st.file_uploader("📁 Загрузите датасет (CSV)", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.success("✅ Данные успешно загружены!")

    else:  # UCI Repository
        if st.button("🌐 Загрузить данные из UCI Repository"):
            try:
                from ucimlrepo import fetch_ucirepo

                with st.spinner("Загрузка данных из UCI Repository..."):
                    # Загрузка датасета AI4I 2020 Predictive Maintenance Dataset
                    dataset = fetch_ucirepo(id=601)
                    data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

                st.success("✅ Данные успешно загружены из UCI Repository!")

            except Exception as e:
                st.error(f"❌ Ошибка при загрузке данных: {str(e)}")
                st.info("💡 Установите библиотеку: pip install ucimlrepo")

    if data is not None:
        # Отображение основной информации о данных
        st.header("📊 Обзор данных")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Количество записей", data.shape[0])
        with col2:
            st.metric("Количество признаков", data.shape[1])
        with col3:
            st.metric("Пропущенные значения", data.isnull().sum().sum())

        # Показать первые строки данных
        st.subheader("🔍 Первые строки данных")
        st.dataframe(data.head())

        # Информация о столбцах
        st.subheader("📋 Информация о столбцах")
        buffer_info = []
        for col in data.columns:
            buffer_info.append({
                "Столбец": col,
                "Тип данных": str(data[col].dtype),
                "Уникальных значений": data[col].nunique(),
                "Пропуски": data[col].isnull().sum()
            })
        st.dataframe(pd.DataFrame(buffer_info))

        # Предобработка данных
        st.header("🔄 Предобработка данных")

        # Создаем копию данных для обработки
        processed_data = data.copy()

        # Удаляем ненужные столбцы
        columns_to_drop = ['UDI', 'Product ID']
        if 'Machine failure' in processed_data.columns:
            # Если есть отдельные типы отказов, удаляем их и оставляем только общий Machine failure
            failure_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
            columns_to_drop.extend([col for col in failure_columns if col in processed_data.columns])
            target_column = 'Machine failure'
        else:
            target_column = 'Target'  # если используется другое название

        # Удаляем столбцы, которые существуют
        columns_to_drop = [col for col in columns_to_drop if col in processed_data.columns]
        if columns_to_drop:
            processed_data = processed_data.drop(columns=columns_to_drop)
            st.write(f"🗑️ Удалены столбцы: {columns_to_drop}")

        # Кодирование категориальной переменной Type
        if 'Type' in processed_data.columns:
            le = LabelEncoder()
            processed_data['Type'] = le.fit_transform(processed_data['Type'])
            st.write("🔤 Категориальная переменная 'Type' преобразована в числовую")

        # Переименуем целевую переменную для единообразия
        if target_column in processed_data.columns and target_column != 'Target':
            processed_data = processed_data.rename(columns={target_column: 'Target'})

        st.write("✅ Предобработка данных завершена")

        # Анализ целевой переменной
        if 'Target' in processed_data.columns:
            st.subheader("🎯 Анализ целевой переменной")

            col1, col2 = st.columns(2)

            with col1:
                target_counts = processed_data['Target'].value_counts()
                st.write("Распределение классов:")
                st.write(target_counts)

                # График распределения
                fig, ax = plt.subplots(figsize=(8, 6))
                target_counts.plot(kind='bar', ax=ax, color=['lightblue', 'lightcoral'])
                ax.set_title('Распределение классов')
                ax.set_xlabel('Класс (0 - норма, 1 - отказ)')
                ax.set_ylabel('Количество')
                plt.xticks(rotation=0)
                st.pyplot(fig)

            with col2:
                # Процентное соотношение
                target_percentage = processed_data['Target'].value_counts(normalize=True) * 100
                st.write("Процентное соотношение:")
                for idx, val in target_percentage.items():
                    st.write(f"Класс {idx}: {val:.2f}%")

        # Корреляционная матрица
        st.subheader("🔗 Корреляционная матрица")

        # Выбираем только числовые столбцы
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
        correlation_matrix = processed_data[numeric_columns].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Корреляционная матрица признаков')
        st.pyplot(fig)

        # Разделение данных и обучение моделей
        if 'Target' in processed_data.columns:
            st.header("🤖 Обучение моделей машинного обучения")

            # Подготовка данных
            X = processed_data.drop(columns=['Target'])
            y = processed_data['Target']

            # Масштабирование признаков
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

            # Разделение на обучающую и тестовую выборки
            test_size = st.sidebar.slider("Размер тестовой выборки (%)", 10, 40, 20)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled_df, y, test_size=test_size/100, random_state=42
            )

            st.write(f"📈 Размер обучающей выборки: {len(X_train)} записей")
            st.write(f"📉 Размер тестовой выборки: {len(X_test)} записей")

            # Выбор моделей для обучения
            st.subheader("🎯 Выбор моделей")

            model_options = {
                "Logistic Regression": LogisticRegression(random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
                "Support Vector Machine": SVC(kernel='linear', probability=True, random_state=42)
            }

            selected_models = st.multiselect(
                "Выберите модели для обучения:",
                list(model_options.keys()),
                default=list(model_options.keys())
            )

            if st.button("🚀 Запустить обучение моделей"):
                if selected_models:
                    results = {}

                    # Прогресс бар
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, model_name in enumerate(selected_models):
                        status_text.text(f"Обучение модели: {model_name}")

                        # Обучение модели
                        model = model_options[model_name]
                        model.fit(X_train, y_train)

                        # Предсказания
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]

                        # Метрики
                        accuracy = accuracy_score(y_test, y_pred)
                        conf_matrix = confusion_matrix(y_test, y_pred)
                        class_report = classification_report(y_test, y_pred)
                        roc_auc = roc_auc_score(y_test, y_pred_proba)

                        # Сохранение результатов
                        results[model_name] = {
                            'model': model,
                            'accuracy': accuracy,
                            'confusion_matrix': conf_matrix,
                            'classification_report': class_report,
                            'roc_auc': roc_auc,
                            'y_pred_proba': y_pred_proba
                        }

                        progress_bar.progress((i + 1) / len(selected_models))

                    status_text.text("✅ Обучение завершено!")

                    # Отображение результатов
                    st.header("📊 Результаты обучения моделей")

                    # Сравнительная таблица метрик
                    metrics_df = pd.DataFrame({
                        'Модель': list(results.keys()),
                        'Accuracy': [results[model]['accuracy'] for model in results.keys()],
                        'ROC-AUC': [results[model]['roc_auc'] for model in results.keys()]
                    }).round(4)

                    st.subheader("🏆 Сравнение моделей")
                    st.dataframe(metrics_df.sort_values('ROC-AUC', ascending=False))

                    # Лучшая модель
                    best_model_name = metrics_df.loc[metrics_df['ROC-AUC'].idxmax(), 'Модель']
                    st.success(f"🥇 Лучшая модель: {best_model_name}")

                    # Детальные результаты для каждой модели
                    for model_name, result in results.items():
                        with st.expander(f"📈 Подробные результаты: {model_name}"):

                            col1, col2 = st.columns(2)

                            with col1:
                                st.write(f"**Accuracy:** {result['accuracy']:.4f}")
                                st.write(f"**ROC-AUC:** {result['roc_auc']:.4f}")

                                # Confusion Matrix
                                st.write("**Матрица ошибок:**")
                                fig, ax = plt.subplots(figsize=(6, 5))
                                sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', 
                                          cmap='Blues', ax=ax)
                                ax.set_title(f'Confusion Matrix - {model_name}')
                                ax.set_xlabel('Predicted')
                                ax.set_ylabel('Actual')
                                st.pyplot(fig)

                            with col2:
                                st.write("**Classification Report:**")
                                st.text(result['classification_report'])

                    # ROC-кривые для всех моделей
                    st.subheader("📊 ROC-кривые")
                    fig, ax = plt.subplots(figsize=(10, 8))

                    for model_name, result in results.items():
                        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
                        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {result['roc_auc']:.3f})")

                    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC-кривые для всех моделей')
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

                    # Интерфейс для предсказания
                    st.header("🔮 Предсказание на новых данных")

                    st.write("Введите значения признаков для предсказания:")

                    with st.form("prediction_form"):
                        col1, col2 = st.columns(2)

                        with col1:
                            if 'Type' in X.columns:
                                type_input = st.selectbox("Type (L=0, M=1, H=2)", [0, 1, 2])
                            air_temp = st.number_input("Air temperature [K]", value=300.0, min_value=250.0, max_value=350.0)
                            process_temp = st.number_input("Process temperature [K]", value=310.0, min_value=250.0, max_value=350.0)

                        with col2:
                            rotational_speed = st.number_input("Rotational speed [rpm]", value=1500, min_value=1000, max_value=3000)
                            torque = st.number_input("Torque [Nm]", value=40.0, min_value=0.0, max_value=100.0)
                            tool_wear = st.number_input("Tool wear [min]", value=100, min_value=0, max_value=300)

                        submit_button = st.form_submit_button("🎯 Сделать предсказание")

                        if submit_button and results:
                            # Подготовка входных данных
                            input_data = {}
                            for col in X.columns:
                                if 'Type' in col:
                                    input_data[col] = type_input
                                elif 'Air' in col or 'temperature' in col.lower():
                                    input_data[col] = air_temp
                                elif 'Process' in col or 'temperature' in col.lower():
                                    input_data[col] = process_temp
                                elif 'speed' in col.lower() or 'rpm' in col.lower():
                                    input_data[col] = rotational_speed
                                elif 'Torque' in col or 'torque' in col.lower():
                                    input_data[col] = torque
                                elif 'wear' in col.lower() or 'Tool' in col:
                                    input_data[col] = tool_wear

                            input_df = pd.DataFrame([input_data])
                            input_scaled = scaler.transform(input_df)

                            st.subheader("🎯 Результаты предсказания")

                            for model_name, result in results.items():
                                model = result['model']
                                prediction = model.predict(input_scaled)[0]
                                prediction_proba = model.predict_proba(input_scaled)[0]

                                with st.expander(f"Результат {model_name}"):
                                    if prediction == 1:
                                        st.error(f"⚠️ Предсказание: ОТКАЗ ОБОРУДОВАНИЯ")
                                    else:
                                        st.success(f"✅ Предсказание: ОБОРУДОВАНИЕ В НОРМЕ")

                                    st.write(f"Вероятность отказа: {prediction_proba[1]:.3f}")
                                    st.write(f"Вероятность нормального состояния: {prediction_proba[0]:.3f}")

                else:
                    st.warning("⚠️ Выберите хотя бы одну модель для обучения")

        else:
            st.error("❌ Целевая переменная 'Target' или 'Machine failure' не найдена в данных")

if __name__ == "__main__":
    analysis_and_model_page()
