import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import date

class DataManager():
    def __init__(self):
        self.circuits = pd.read_csv('data/circuits.csv')
        self.constructor_results = pd.read_csv('data/constructor_results.csv')
        self.constructor_standings = pd.read_csv('data/constructor_standings.csv')
        self.constructors = pd.read_csv('data/constructors.csv')
        self.driver_standings = pd.read_csv('data/driver_standings.csv')
        self.drivers = pd.read_csv('data/drivers.csv')
        self.lap_times = pd.read_csv('data/lap_times.csv')
        self.pit_stops = pd.read_csv('data/pit_stops.csv')
        self.qualifying = pd.read_csv('data/qualifying.csv')
        self.races = pd.read_csv('data/races.csv')
        self.results = pd.read_csv('data/results.csv')
        self.seasons = pd.read_csv('data/seasons.csv')
        self.status = pd.read_csv('data/status.csv')
        self.weather = pd.read_csv('data/weather.csv')
        self.regData = pd.DataFrame()
        self.qualyRegData = pd.DataFrame()
        
    def getRaceData(self, raceId):
        lap_times = self.lap_times.loc[lambda lap_time: lap_time['raceId'] == raceId]
        race = self.races.loc[lambda race: race['raceId'] == raceId]
        results = self.results.loc[lambda result: result['raceId'] == raceId]
        drivers = self.drivers.merge(results, on='driverId')[['driverId', 'code', 'constructorId']]
        lap_times = lap_times.merge(drivers, on='driverId')[['raceId', 'driverId', 'lap', 'milliseconds', 'code', 'constructorId']]
        return lap_times

    def getRaceResults(self, raceId, kind='df'):
        results = self.results.loc[lambda res: res['raceId'] == raceId][['driverId', 'grid', 'positionOrder', 'positionText']]
        results = pd.merge(results, self.drivers, how='inner', on='driverId').drop(['number', 'code', 'dob', 'nationality', 'url'], axis=1)
        if kind == 'json':
            return {'results': list(results.T.to_dict().values())}
        return results
        
    def getRaceDetails(self, raceId, kind='df'):
        race = self.races.loc[lambda r: r['raceId'] == raceId].drop(['time'], axis=1)
        race = pd.merge(race, self.circuits, how='inner', on='circuitId').drop(['circuitRef', 'location', 'lat', 'lng', 'alt'], axis=1)
        weather = self.weather.loc[lambda r: r['raceId'] == raceId].drop(['year', 'round', 'weather_warm', 'weather_cold', 'weather_dry', 'weather_wet', 'weather_cloudy'], axis=1)
        race = pd.merge(race, weather, how='inner', on='raceId').drop(['Unnamed: 0'], axis=1)
        if kind == 'json':
            return {'race': list(race.T.to_dict().values())}
        return race

    def getPastRaces(self):
        races = self.races.loc[lambda race: race['year'] >= 2020]
        for i, row in races.iterrows():
            if row['date'] > str(date.today()):
                races.drop(index=i, inplace=True)
        return {'races': list(races.T.to_dict().values())}

    def getFutureRaces(self, form='df'):
        races = self.races.loc[lambda race: race['year'] >= 2020]
        for i, row in races.iterrows():
            if row['date'] < str(date.today()):
                races.drop(index=i, inplace=True)
        if form == 'json':
            return {'races': list(races.T.to_dict().values())}
        else:
            return races

    def generateQualyRegData(self):
        res = self.qualifying.drop(['number', 'qualifyId', 'q1', 'q2', 'q3'], axis=1)
        res = pd.merge(res, self.races, how='inner', on=['raceId']).drop(['name', 'time', 'url'], axis=1)
        for i, row in res.iterrows():
            if row['date'] > str(date.today()):
                res.drop(index=i, inplace=True)
        res = res.loc[lambda r: r['year'] >= 2005]
        res = res.drop(['date'], axis=1)
        self.qualyRegData = res
        
    def getQualyRegData(self):
        if self.qualyRegData.empty:
            self.generateQualyRegData()
        return self.qualyRegData

    def generateRegData(self):
        res = self.results.drop(['number', 'position', 'positionText', 'points', 'laps', 'time', 'milliseconds', 'fastestLap', 'rank', 'fastestLapTime', 'fastestLapSpeed', 'statusId'], axis=1)
        res = pd.merge(res, self.races, how='inner', on=['raceId']).drop(['name', 'time', 'url'], axis=1)
        for i, row in res.iterrows():
            if row['date'] > str(date.today()):
                res.drop(index=i, inplace=True)
        res = res.loc[lambda r: r['year'] >= 2005]
        res = pd.merge(res, self.weather, how='inner', on=['raceId', 'year', 'round']).drop(['weather'], axis=1)
        res = res.drop(['date', 'Unnamed: 0'], axis=1)
        self.regData = res
        
    def getRegData(self):
        if self.regData.empty:
            self.generateRegData()
        return self.regData
        
    def getSeasonDrivers(self, season):
        races = self.races.loc[lambda res: res['year'] == season]
        upperId = max(races['raceId'])
        lowerId = min(races['raceId'])
        results = self.results.loc[lambda res: res['raceId'] >= lowerId]
        results = results.loc[lambda res: res['raceId'] <= upperId]
        drivers = results['driverId'].unique()
        drivers = pd.DataFrame(drivers, columns=['driverId'])
        drivers = pd.merge(drivers, self.drivers, how='inner', on=['driverId'])
        res_unique = results.drop_duplicates('driverId', keep='first')
        drivers = pd.merge(drivers, res_unique, how='inner', on=['driverId'])
        return drivers[['driverId', 'driverRef', 'constructorId']]