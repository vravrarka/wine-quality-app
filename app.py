import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')

@st.cache_data
def load_test_data():
    return pd.read_csv(APP_DIR / 'test.csv')

@st.cache_data
def predict_test_proba(_model, features):
    return _model.predict_proba(features)[:, 1]

model = load_model()
test_df = load_test_data()

feature_cols = [col for col in test_df.columns if col != 'target']
X_test = test_df[feature_cols]
y_test = test_df['target']
p_test = predict_test_proba(model, X_test)

config = json.loads(
    (APP_DIR / 'config.json').read_text(encoding='utf-8')
)

threshold = st.sidebar.slider(
    'Порог рекомендации',
    min_value=0.0,
    max_value=1.0,
    value=float(config['threshold']),
    step=0.01
)

st.sidebar.caption('Один порог используется во всех вкладках.')

st.title('  Рекомендации вин')
st.write('Оценка вероятности хорошего вина по лабораторным характеристикам.')

tab1, tab2, tab3, tab4 = st.tabs([
    'Одно вино',
    'Модельные метрики',
    'Пакетное предсказание',
    'Коэффициенты'
])

labels = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный SO₂',
    'total_sulfur_dioxide': 'Общий SO₂',
    'density': 'Плотность',
    'pH': 'pH',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь'
}

with tab1:
    st.subheader('Характеристики вина')
    entered = {}
    cols = st.columns(2)
    numeric_features = [
        col for col in feature_cols if col != 'is_red'
    ]

    for i, col in enumerate(numeric_features):
        low = float(test_df[col].min())
        high = float(test_df[col].max())
        middle = float(test_df[col].median())
        with cols[i % 2]:
            entered[col] = st.number_input(
                labels.get(col, col),
                min_value=low,
                max_value=high,
                value=middle,
                step=max((high - low) / 100, 0.0001),
                format='%.4f',
                key=f'input_{col}'
            )

    entered['is_red'] = int(
        st.toggle('Красное вино (выкл. — белое)', value=False)
    )

    wine = pd.DataFrame([entered], columns=feature_cols)
    prob = float(model.predict_proba(wine)[0, 1])
    st.metric('Вероятность хорошего вина', f'{prob:.1%}')
    if prob >= threshold:
        st.success('Рекомендация: хорошее вино')
    else:
        st.warning('Рекомендация: не включать в подборку')

with tab2:
    st.subheader('Метрики на тестовой выборке')
    pred = (p_test >= threshold).astype(int)
    prec = metrics.precision_score(
        y_test, pred, zero_division=0
    )
    rec = metrics.recall_score(
        y_test, pred, zero_division=0
    )
    f1 = metrics.f1_score(
        y_test, pred, zero_division=0
    )
    a, b, c = st.columns(3)
    a.metric('Precision', f'{prec:.3f}')
    b.metric('Recall', f'{rec:.3f}')
    c.metric('F1', f'{f1:.3f}')
    cm = metrics.confusion_matrix(
        y_test, pred, labels=[0, 1]
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    metrics.ConfusionMatrixDisplay(
        cm, display_labels=[0, 1]
    ).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Матрица ошибок')
    st.pyplot(fig)
    plt.close(fig)
    fpr, tpr, _ = metrics.roc_curve(y_test, p_test)
    fpr_point = cm[0, 1] / cm[0].sum()
    fig, ax = plt.subplots()
    ax.plot(
        fpr, tpr,
        label=f'ROC-AUC = {metrics.roc_auc_score(y_test, p_test):.3f}'
    )
    ax.plot(
        [0, 1], [0, 1], 'k--',
        label='Случайная модель'
    )
    ax.scatter(
        fpr_point, rec, s=85, zorder=5,
        label=f'Порог = {threshold:.2f}'
    )
    ax.set(xlabel='FPR', ylabel='TPR', title='ROC-кривая')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    pr_precision, pr_recall, _ = (
        metrics.precision_recall_curve(y_test, p_test)
    )
    fig, ax = plt.subplots()
    ax.plot(
        pr_recall, pr_precision,
        label=f'AP = {metrics.average_precision_score(y_test, p_test):.3f}'
    )
    ax.axhline(
        y_test.mean(),
        linestyle='--',
        color='gray',
        label='Случайная модель'
    )
    ax.scatter(
        rec, prec if pred.sum() else 1.0,
        s=85, zorder=5,
        label=f'Порог = {threshold:.2f}'
    )
    ax.set(
        xlabel='Recall',
        ylabel='Precision',
        title='PR-кривая',
        xlim=(0, 1),
        ylim=(0, 1)
    )
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        'При снижении порога в подборку попадёт больше хороших вин, '
        'но и больше посредственных. При повышении порога — наоборот.'
    )

with tab3:
    st.subheader('Загрузка CSV')
    uploaded = st.file_uploader(
        'Загрузите файл с 12 признаками (без target)',
        type='csv'
    )

    if uploaded is not None:
        try:
            incoming = pd.read_csv(uploaded)
            missing = [
                col for col in feature_cols
                if col not in incoming.columns
            ]
            if missing:
                st.error(
                    'Не хватает столбцов: ' + ', '.join(missing)
                )
            elif incoming.empty:
                st.error('CSV-файл пуст.')
            else:
                data = incoming[feature_cols].apply(
                    pd.to_numeric, errors='raise'
                )
                if data.isna().any().any():
                    st.error('В признаках есть пустые значения.')
                else:
                    probabilities = model.predict_proba(data)[:, 1]
                    result = incoming.copy()
                    result['probability_good'] = probabilities
                    result['prediction'] = (
                        probabilities >= threshold
                    ).astype(int)
                    st.dataframe(result)
                    st.download_button(
                        'Скачать предсказания',
                        result.to_csv(index=False).encode('utf-8-sig'),
                        file_name='wine_predictions.csv',
                        mime='text/csv'
                    )
        except (
            ValueError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
            UnicodeError
        ) as exc:
            st.error(f'Не удалось обработать CSV: {exc}')

with tab4:
    st.subheader('Влияние признаков')
    clf = model.named_steps['logisticregression']
    coefficients = pd.Series(
        clf.coef_[0], index=feature_cols
    ).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(coefficients.index, coefficients.values)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Коэффициент после StandardScaler')
    ax.set_title('Коэффициенты логистической регрессии')
    st.pyplot(fig)
    plt.close(fig)
    st.write(
        'Положительные коэффициенты повышают оценку '
        'вероятности хорошего вина, отрицательные — понижают '
        'при прочих равных.'
    )
    st.caption(
        f'Сильнее всего за: {coefficients.idxmax()}. '
        f'Сильнее всего против: {coefficients.idxmin()}.'
    )