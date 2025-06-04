import pickle

import numpy as np
import streamlit as st

input_gravity = 1.018
input_ph = 6.02
input_osmo = 612.84
input_cond = 20.8
input_urea = 226.4
input_calc = 4.13


def main():
    global input_gravity, input_ph, input_osmo, input_cond, input_urea, input_calc

    st.set_page_config(
        page_title='Kidney stone prediction',
        page_icon='🪨',
        layout='wide'
    )

    st.title("🪨 Kidney Stone Prediction")

    main_cols = st.columns([0.55, 0.05, 0.4])

    main_cols[0].write("""
### Project Overview

* This project predicts the likelihood of kidney stones using urine analysis data.

* It uses a dataset with clinical features such as pH, gravity, calcium levels, and others.

* Data is cleaned, outliers removed, and new features like acidity/alkalinity are engineered.

* Numerical features are scaled using MinMaxScaler for better model performance.

* 20+ machine learning models (Logistic Regression, SVM, Random Forest, etc.) are trained.

* A caching system stores trained models to avoid redundant training.

* Each model is evaluated using accuracy, recall, precision, F1-score, and ROC-AUC.

* Top-performing models like Random Forest and AdaBoost are visualized using decision trees.

* Evaluation results are stored in a unified DataFrame and plotted for comparison.

* The best model can be saved for deployment in clinical decision support systems.
""")

    main_cols[2].image(r'assets/kidney_cover.png', width=500)

    st.divider()

    st.header("Model")

    cols = st.columns([0.55, 0.05, 0.4])

    with cols[0]:
        st.write("""
***specific gravity*** , the density of the urine relative to water; 

***pH***, the negative logarithm of the hydrogen ion; 

***osmolarity (mOsm)***, a unit used in biology and medicine but not in
physical chemistry. Osmolarity is proportional to the concentration of
molecules in solution; 

***conductivity (mMho milliMho)***. One Mho is one
reciprocal Ohm. Conductivity is proportional to the concentration of charged
ions in solution; 

***urea concentration*** in millimoles per litre; and 

***calcium concentration*** in millimolesllitre.
        """.strip())

        grid = [st.columns(2) for x in range(4)]

        input_gravity = grid[0][0].number_input("Specific Gravity", min_value=1.0, max_value=1.1, value=input_gravity)
        input_ph = grid[0][1].number_input("pH", min_value=0.0, max_value=14.0, value=input_ph)
        input_osmo = grid[1][0].number_input("Osmolarity (mOsm)", min_value=180.0, max_value=1300.0, value=input_osmo)
        input_cond = grid[1][1].number_input("Conductivity (mMho milliMho)", min_value=5.0, max_value=40.0, step=1.0,
                                             value=input_cond)
        input_urea = grid[2][0].number_input("urea concentration (millimoles per litre)", min_value=10.0,
                                             max_value=620.0,
                                             step=10.0, value=input_urea)
        input_calc = grid[2][1].number_input("calcium concentration (milli molesl litre)", min_value=0.10,
                                             max_value=15.0,
                                             step=0.1, value=input_calc)


        st.warning("""This kidney stone prediction model is developed for educational and research purposes. 

It is not intended for medical diagnosis or treatment. 

Please consult a qualified healthcare professional for any medical concerns.""")

        st.container(height=10, border=False)

        if st.button('Predict', type='primary', use_container_width=True, icon='❔'):
            output_label = predictTarget()

            with st.container(border=True):
                st.subheader("Output:")
                if output_label < 0.5:
                    original_title = '<p style="font-family:sans-serif; color:Green; font-size: 20px;">Negative, No stone detected</p>'
                    st.markdown(original_title, unsafe_allow_html=True)
                else:
                    original_title = '<p style="font-family:sans-serif; color:Red; font-size: 20px;">Positive, There are a stone detected</p>'
                    st.markdown(original_title, unsafe_allow_html=True)
    with cols[2]:
        st.image(r'assets/stone.png', width=500)


def load_obj(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_resources():
    MIN_MAX_SCALER_PATH = r'encoders_scalers/min-max-scaler.pickle'
    MODEL_PATH = r'models_cache/decision tree max-depth=5.pickle'

    min_max_scaler = load_obj(MIN_MAX_SCALER_PATH)
    model = load_obj(MODEL_PATH)

    return min_max_scaler, model


def predictTarget():
    global input_gravity, input_ph, input_osmo, input_cond, input_urea, input_calc

    min_max_scaler, model = load_resources()

    H_ion = 10 ** (-input_ph)
    acidic_0_alkaline_1 = int(input_ph < 7.0)
    calc_osmo = input_calc * input_osmo
    cond_osmo_ratio = input_cond / input_osmo
    urea_gravity_ratio = input_urea / input_gravity

    X = np.array([
        input_gravity,
        input_ph,
        input_osmo,
        input_cond,
        input_urea,
        input_calc,
        H_ion,
        acidic_0_alkaline_1,
        calc_osmo,
        cond_osmo_ratio,
        urea_gravity_ratio
    ]).reshape((1, -1))

    X[:, 0:-3] = min_max_scaler.transform(X[:, 0:-3])

    Y = model.predict(X)[0]

    return Y


if __name__ == "__main__":
    main()
