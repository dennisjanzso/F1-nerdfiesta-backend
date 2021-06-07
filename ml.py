from __future__ import annotations
import visualizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from abc import ABC, abstractmethod
from datetime import date, datetime
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import joblib

class DriverModel():
    def __init__(self, DM, driverId, driver_name):
        self.DM = DM
        self.driverId = driverId
        self.name = driver_name
        self.model = RandomForestRegressor()
        self.error = 0
        self.r2 = 0
        self.generateDfs()
        
    @abstractmethod
    def generateDfs(self):
        pass
        
    def genModel(self, n):
        print('Building model for ', self.name)
        for i in range(0, n):
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=0.9)
            regr = RandomForestRegressor()
            regr.fit(X_train, y_train.values.ravel())
            y_pred = regr.predict(X_test)
            error = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            if r2 > self.r2:
                self.model = regr
                self.r2 = r2
                self.error = error
        print('Model for ', self.name, ' completed')
        
    def getModelScore(self):
        return (self.error, self.r2)
    
    def getValidModel(self):
        return not (self.error == 0 and self.r2 == 0)
    
    def getPrediction(self, *args, **kwargs):
        pass

class DriverQualyModel(DriverModel):
    def generateDfs(self):
        self.df = self.DM.getQualyRegData().loc[lambda res: res['driverId'] == self.driverId]
        self.df = pd.get_dummies(self.df, columns=['constructorId', 'year', 'circuitId'])
        self.X = self.df.drop(['position', 'raceId', 'driverId'], axis=1)
        self.y = self.df[['position']]
        self.mode = 'q'
        
    def getPrediction(self, rnd, constructorId, year, circuitId):
        pred_values = {}
        pred_values['round'] = rnd
        pred_values['constructorId_' + str(constructorId)] = 1
        pred_values['year_' + str(year)] = 1
        pred_values['circuitId_' + str(circuitId)] = 1
        pred_vector = []
        for col in self.X.columns:
            if col in pred_values.keys():
                pred_vector.append(pred_values[col])
            else:
                pred_vector.append(0)
        pred_df = pd.DataFrame([pred_vector], columns=self.X.columns)
        return self.model.predict(pred_df)

class DriverRaceModel(DriverModel):
    def generateDfs(self):
        self.df = self.DM.getRegData().loc[lambda res: res['driverId'] == self.driverId]
        self.df = pd.get_dummies(self.df, columns=['constructorId', 'year', 'circuitId'])
        self.X = self.df.drop(['positionOrder', 'resultId', 'raceId', 'driverId'], axis=1)
        self.y = self.df[['positionOrder']]
        self.mode = 'r'
        
    def getPrediction(self, grid, rnd, weather, constructorId, year, circuitId):
        pred_values = {}
        pred_values['grid'] = grid
        pred_values['round'] = rnd
        pred_values['weather_' + weather] = 1
        pred_values['constructorId_' + str(constructorId)] = 1
        pred_values['year_' + str(year)] = 1
        pred_values['circuitId_' + str(circuitId)] = 1
        pred_vector = []
        for col in self.X.columns:
            if col in pred_values.keys():
                pred_vector.append(pred_values[col])
            else:
                pred_vector.append(0)
        pred_df = pd.DataFrame([pred_vector], columns=self.X.columns)
        return self.model.predict(pred_df)

class RegModels():
    def __init__(self, DM):
        self.DM = DM
        self.df = self.DM.getRegData()
        self.currentDrivers = self.DM.getSeasonDrivers(2021)
        self.race_models = {}
        self.qualy_models = {}
        
    def buildRaceModels(self, n):
        for i, driver in self.currentDrivers.iterrows():
            self.race_models[driver['driverRef']] = DriverRaceModel(self.DM, driver['driverId'], driver['driverRef'])
            self.race_models[driver['driverRef']].genModel(n)
            print(self.race_models[driver['driverRef']].getModelScore())
            print('Is model valid: ', self.race_models[driver['driverRef']].getValidModel())
        
        
    def buildQualyModels(self, n):
        for i, driver in self.currentDrivers.iterrows():
            self.qualy_models[driver['driverRef']] = DriverQualyModel(self.DM, driver['driverId'], driver['driverRef'])
            self.qualy_models[driver['driverRef']].genModel(n)
            print(self.qualy_models[driver['driverRef']].getModelScore())
            print('Is model valid: ', self.qualy_models[driver['driverRef']].getValidModel())
            
    def predictQualy(self, rnd, year, circuitId):
        pred_scores = {}
        for i, row in self.currentDrivers.iterrows():
            print('predicting for ', row['driverRef'])
            if self.qualy_models[row['driverRef']].getValidModel():
                score = self.qualy_models[row['driverRef']].getPrediction(rnd, row['constructorId'], year, circuitId)
                pred_scores[row['driverRef']] = score.tolist()[0]
            else:
                pred_scores[row['driverRef']] = float("NaN")
        p_df = pd.DataFrame()
        p_df = p_df.from_dict(pred_scores, orient='index', columns=['pred_q_score'])
        p_df['driverRef'] = p_df.index
        p_df.sort_values(by='pred_q_score', inplace=True)
        p_df.reset_index(inplace=True, drop=True)
        p_df = pd.merge(p_df, self.currentDrivers, how='inner', on=['driverRef'])
        return p_df
            
            
    def predictRace(self, drivers, rnd, weather, year, circuitId):
        pred_scores = {}
        for i, row in drivers.iterrows():
            if self.race_models[row['driverRef']].getValidModel():
                score = self.race_models[row['driverRef']].getPrediction(i+1, rnd, weather, row['constructorId'], year, circuitId)
                pred_scores[row['driverRef']] = score.tolist()[0]
            else:
                pred_scores[row['driverRef']] = float("NaN")
        p_df = pd.DataFrame()
        p_df = p_df.from_dict(pred_scores, orient='index', columns=['pred_r_score'])
        p_df['driverRef'] = p_df.index
        p_df.sort_values(by='pred_r_score', inplace=True)
        p_df.reset_index(inplace=True, drop=True)
        p_df = pd.merge(p_df, self.currentDrivers, how='inner', on=['driverRef'])
        return p_df
    
    def getTotalError(self):
        error = 0
        for model in self.qualy_models.keys():
            error += self.qualy_models[model].getModelScore()[0]
        for model in self.race_models.keys():
            error += self.race_models[model].getModelScore()[0]
        return error

class Predictor():
    def __init__(self, data_manager, build_n=1):
        self.DM = data_manager
        self.RM = RegModels(self.DM)
        self.Viz = visualizer.Visualizer(self.DM, visualizer.PredResultPlotter())
        self.initialize_models(build_n)
        
    def initialize_models(self, build_n):
        self.RM.buildQualyModels(build_n)
        self.RM.buildRaceModels(build_n)
        
    def predict_race(self, raceId, weather, save_name):
        races = self.DM.getFutureRaces()
        race = races.loc[lambda r: r['raceId'] == raceId].iloc[0]
        qualy_pred = self.RM.predictQualy(race['round'], race['year'], race['circuitId'])
        self.Viz.execute_strategy(df=qualy_pred, kind='q', savename='q_'+save_name)
        race_pred = self.RM.predictRace(qualy_pred, race['round'], weather, race['year'], race['circuitId'])
        self.Viz.execute_strategy(df=race_pred, kind='r', savename='r_'+save_name)
        return {'ticket': save_name,
            'race': list(race_pred.T.to_dict().values()),
            'qualy': list(qualy_pred.T.to_dict().values())}