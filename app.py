from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from src.blueberry.a_constants import *
from src.blueberry.f_utils.common import load_pickle, read_yaml
from pathlib import Path


app = Flask(__name__)

candidates = {}


@app.route('/')
def index():
    return render_template('Intro.html')


@app.route('/data')
def data():
    return render_template('data.html')


@app.route('/eda/univariate')
def univariate():
    return render_template('univariate.html')


@app.route('/eda/bivariate')
def bivariate():
    return render_template('bivariate.html')


@app.route('/model/modelanalysis')
def modelanalysis():
    return render_template('modelanalysis.html')


def convert_to_float(value):
    return float(value) if value is not None and value != '' else None


@app.route('/model/modelplots')
def modelplots():
    return render_template('modelplots.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        # Retrieve the input data from the form
        clonesize = request.form.get('clonesize')
        honeybee = request.form.get('honeybee')
        bumbles = request.form.get('bumbles')
        andrena = request.form.get('andrena')
        osmia = request.form.get('osmia')
        MaxOfUpperTRange = request.form.get('MaxOfUpperTRange')
        MinOfUpperTRange = request.form.get('MinOfUpperTRange')
        AverageOfUpperTRange = request.form.get('AverageOfUpperTRange')
        MaxOfLowerTRange = request.form.get('MaxOfLowerTRange')
        MinOfLowerTRange = request.form.get('MinOfLowerTRange')
        AverageOfLowerTRange = request.form.get('AverageOfLowerTRange')
        RainingDays = request.form.get('RainingDays')
        AverageRainingDays = request.form.get('AverageRainingDays')
        fruitset = request.form.get('fruitset')
        fruitmass = request.form.get('fruitmass')
        seeds = request.form.get('seeds')

        # Load the saved list of models using pickle
        # Change this to the best model for blueberry yield
        best_model_name = 'xgb_regressor'
        pickle_file_path = Path(
            "artifacts/model_trainer/trained_models.joblib")
        loaded_models = load_pickle(pickle_file_path)

        rf_models = loaded_models[best_model_name]

        # Convert form data to float
        def convert_to_float(value):
            return float(value) if value is not None and value != '' else None

        clonesize = convert_to_float(clonesize)
        honeybee = convert_to_float(honeybee)
        bumbles = convert_to_float(bumbles)
        andrena = convert_to_float(andrena)
        osmia = convert_to_float(osmia)
        MaxOfUpperTRange = convert_to_float(MaxOfUpperTRange)
        MinOfUpperTRange = convert_to_float(MinOfUpperTRange)
        AverageOfUpperTRange = convert_to_float(AverageOfUpperTRange)
        MaxOfLowerTRange = convert_to_float(MaxOfLowerTRange)
        MinOfLowerTRange = convert_to_float(MinOfLowerTRange)
        AverageOfLowerTRange = convert_to_float(AverageOfLowerTRange)
        RainingDays = convert_to_float(RainingDays)
        AverageRainingDays = convert_to_float(AverageRainingDays)
        fruitset = convert_to_float(fruitset)
        fruitmass = convert_to_float(fruitmass)
        seeds = convert_to_float(seeds)

        # Create a dictionary from the form data
        data = {
            'clonesize': [clonesize],
            'honeybee': [honeybee],
            'bumbles': [bumbles],
            'andrena': [andrena],
            'osmia': [osmia],
            'MaxOfUpperTRange': [MaxOfUpperTRange],
            'MinOfUpperTRange': [MinOfUpperTRange],
            'AverageOfUpperTRange': [AverageOfUpperTRange],
            'MaxOfLowerTRange': [MaxOfLowerTRange],
            'MinOfLowerTRange': [MinOfLowerTRange],
            'AverageOfLowerTRange': [AverageOfLowerTRange],
            'RainingDays': [RainingDays],
            'AverageRainingDays': [AverageRainingDays],
            'fruitset': [fruitset],
            'fruitmass': [fruitmass],
            'seeds': [seeds],
        }

        df = pd.DataFrame(data)
        preds = [model.predict(np.array(df)) for model in rf_models]
        print(preds)
        preds_mean = sum(preds) / len(preds)

        return render_template('predict.html', predicted_blueberry_yield=preds_mean[0])

    except Exception as e:
        return render_template('predict.html', error_message=str(e))


@app.route('/form')
def show_form():
    return render_template('predict.html', preds_final=None, error_message=None)


if __name__ == '__main__':
    app.run(debug=True)
