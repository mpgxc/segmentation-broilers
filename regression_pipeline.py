#!/usr/bin/env python
# coding: utf-8

import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm 
from glob import glob
from skimage.io import imread
from skimage.measure import label, regionprops_table
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, ElasticNet, SGDRegressor
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')


def print_evaluate(true, predicted):  
    
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    
def run_all_regressions(X_train, X_test, Y_train, Y_test, regs):

    for name, model in regs.items():

        model.fit(X_train, Y_train)
        
        print(f'\n-----{name}------')
        print('[Train] -------------')
        print_evaluate(Y_train, model.predict(X_train))

        print('[Test] --------------')
        print_evaluate(Y_test, model.predict(X_test))
        
def extract_props_from_image(mask):
    
    label_image = label(mask)

    feature_names = ['bbox_area',
                     'solidity',
                     'equivalent_diameter', 
                     'orientation',
                     'convex_area', 
                     'area',
                     'extent',
                     'eccentricity',
                     'major_axis_length',
                     'feret_diameter_max',
                     'perimeter',
                     'minor_axis_length']
    
    return pd.DataFrame(regionprops_table(label_image, properties=feature_names))

def build_features_dataset(origin_images):

    out = pd.DataFrame()

    for first_mask in tqdm(glob(origin_images + "/*")):
        
        try:
            mask = imread(first_mask)

            props = extract_props_from_image(mask)
            label_name = first_mask.split('/')[-1].split('.')[0]
            props['label'] = int(label_name)

            out = pd.concat([out, props], ignore_index=True)
        
        except Exception as e:
            print("Erro na imagem:", e, end="\n")
            break
            
    return out

def labeling_dataset(pd_features, weights_filepath):

    pd_dataset = pd_features.copy()
    
    broilers_weights = pd.read_csv(weights_filepath)

    pd_dataset['target'] = -1

    for anilha, peso in zip(broilers_weights.anilhas,  broilers_weights.pesos):

        try: 
            index, *_ = pd_dataset[pd_dataset.label == anilha].index
            pd_dataset.target.iloc[index] = peso
        except:
            continue

    pd_dataset.label.apply(lambda value: value in broilers_weights.anilhas.to_list())

    pd_dataset.drop(pd_dataset[pd_dataset.target < 1].index, inplace=True)

    return pd_dataset

if __name__ == "__main__":

    features = build_features_dataset("auto_selected_masks")
    features_labeling = labeling_dataset(features, "galinhas_pesos.csv")

    ### Rodando a Regression
    corrmat = features_labeling.corr()

    features_names = list(dict(corrmat[corrmat.target >= 0].target).keys())

    X = features_labeling[features_names].drop(labels=['target'], axis=1)
    y = features_labeling.target

    X_train, X_test, y_train, y_test = train_test_split(X.values, 
                                                        y.values, 
                                                        test_size=0.2,
                                                        random_state=0)

    scaler = StandardScaler()

    scaled_x_train = scaler.fit_transform(X_train)
    scaled_x_test = scaler.transform(X_test)


    run_all_regressions(scaled_x_train, 
                        scaled_x_test,
                        y_train,
                        y_test,
                        regs = {
                            "Lasso": Lasso(), 
                            "LinearRegression": LinearRegression(),
                        })

    print('=' * 50)

    run_all_regressions(scaled_x_train, 
                        scaled_x_test,
                        y_train,
                        y_test,
                        regs = {
                            "Ridge": Ridge(),
                            "BayesianRidge": BayesianRidge(), 
                            "ElasticNet": ElasticNet(), 
                            "SGDRegressor": SGDRegressor()
                        })

    print('=' * 50)

    print("Running C V Model...")

    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(X)

    model = SGDRegressor()
    scores = cross_val_score(model, scaled_x, y, cv=5, scoring='r2')

    print("Todos Scores: ", scores)
    print("R2 Score Médio: ", np.mean(scores))

