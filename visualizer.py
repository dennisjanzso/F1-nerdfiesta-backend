from __future__ import annotations
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from abc import ABC, abstractmethod
from matplotlib.patches import Rectangle
import math
from sklearn.cluster import KMeans

team_colors = {
    131: '#00D2BE',
    6: '#DC0000',
    9: '#0600EF',
    214: '#0090FF',
    210: '#5e5e5e',
    117: '#006F62',
    213: '#2B4562',
    1: '#FF8700',
    51: '#900000',
    3: '#005AFF',
    4: '#FFF500',
    211: '#F596C8'}


class Visualizer():
    def __init__(self, data_manager, strategy: Strategy) -> None:
        self._strategy = strategy
        self.data_manager = data_manager

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        self._strategy = strategy

    def execute_strategy(self, *args, **kwargs) -> None:
        self._strategy.do_algorithm(self.data_manager, *args, **kwargs)

class Strategy(ABC):
    
    @abstractmethod
    def do_algorithm(self, data_manager, *args, **kwargs):
        pass

class RacePlotter(Strategy):
    def do_algorithm(self, data_manager, raceId, driver_filter=[], save_name=''):
        plot_sc = False
        
        # Set up plt
        sns.set_theme(style="ticks")
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # Set up data
        df = data_manager.getRaceData(raceId)
        if len(driver_filter) == 0:
            ds = df.groupby(['constructorId'], axis=0)['driverId'].unique()
            drivers = []
            for d in ds:
                try:
                    drivers.append(d[0])
                    drivers.append(d[1])
                except:
                    continue
        else:
            drivers = driver_filter
        time_series = df.pivot(index='lap', columns='driverId', values='milliseconds')
        displayed_constructors = []
        
        # Meansumcum and so on for plottable time series
        time_series['mean_sum'] = time_series.mean(1).cumsum()
        mean_time_series = time_series[['mean_sum']]
        for col in time_series.columns:
            time_series[col] = time_series[col].cumsum()
            time_series[col] = (time_series['mean_sum']-time_series[col])
        time_series = time_series.drop(['mean_sum'], axis=1)
        
        # Plot the drivers
        for driverId in drivers:
            driver = df.loc[lambda lap_time: lap_time['driverId'] == driverId].iloc[0]
            
            line_style = '-'
            if driver['constructorId'] in displayed_constructors:
                line_style = '--'
            else:
                displayed_constructors.append(driver['constructorId'])
            
            sns.lineplot(data=time_series[driverId], ax=ax, label=driver['code'], linestyle=line_style, color=team_colors[driver['constructorId']])
            
        # Fix up plot
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax.get_yaxis().set_visible(False)
        plt.xticks(np.arange(1, len(mean_time_series.index), 1))
        ax.xaxis.grid(True, linestyle='-.')
        ax.set_xlabel('Laps')
        ax.set(frame_on=False)
        xs = np.linspace(len(time_series.index), len(time_series.index)+2, 3)
        ys = np.linspace(time_series.min().min(), time_series.max().max(), 21)
        w, h = xs[1] - xs[0], ys[1] - ys[0]
        for i, x in enumerate(xs[:-1]):
            for j, y in enumerate(ys[:-1]):
                if i % 2 == j % 2:
                    ax.add_patch(Rectangle((x, y), w, h, fill=True, color='#000000'))
        
        # Plot safety cars
        if plot_sc:
            mean_time_series = mean_time_series.diff()
            treshold = mean_time_series.mean(0).values[0]*1.18
            mean_time_series['sc'] = mean_time_series[mean_time_series['mean_sum'] > treshold]
            scs = []
            sca = False
            for i, r in mean_time_series.iterrows():
                if not np.isnan(r['sc']):
                    if not sca:
                        start = i
                        sca = True
                else:
                    if sca:
                        scs.append((start, i))
                        sca = False
            for sc in scs:
                ax.add_patch(Rectangle((sc[0], time_series.min().min()), sc[1]-sc[0], time_series.max().max()+abs(time_series.min().min()), fill=True, color='#ffe730', alpha=0.15))
        
        fig.savefig('cache/' + save_name + '.png', dpi = 300)

class PredResultPlotter(Strategy):
    def do_algorithm(self, data_manager, df, kind='r', savename='predres'):
        fig = plt.figure(figsize=(18, 6))
        ax = fig.gca()
        
        ffont = {'fontname':'monospace'}
        
        for i, row in df.iterrows():
            if not math.isnan(row['pred_'+kind+'_score']):
                ax.bar(x=i+1, height=row['pred_'+kind+'_score'], width=1, color=team_colors[row['constructorId']])
                ax.text(i+0.8, 0.3, row['driverRef'], fontsize=15, color='black', **ffont, rotation=90)
            else:
                ax.text(i+0.8, 0.3, row['driverRef'] + ' - not enough data', fontsize=10, color='black', **ffont, rotation=90)
                
        x_ticks = np.arange(1, 21, 1)
        x_labels = x_ticks.tolist()
        plt.xticks(x_ticks, x_labels)
        
        fig.savefig('cache/'+savename+'.png', dpi = 300)
        
class KmeansPlotter(Strategy):
    def do_algorithm(self, data_manager, driverId, k=3, savename='kmeans'):
        plt.clf()
        driver_res = data_manager.getDriverResults(driverId)[['grid', 'positionOrder']]

        distortions = []
        K = range(1,10)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(driver_res)
            distortions.append(kmeanModel.inertia_)

        k = 1
        changes = []
            
        for i in range(0, len(distortions)):
            try:
                t = abs(distortions[i]/distortions[i+1])
                print(i+1, ' t: ', t)
                changes.append(t)
            except:
                pass
        
        avg = sum(changes)/len(changes)
        
        print(avg)
        
        for i in range(0, len(changes)):
            if changes[i] < avg:
                k=i+1
                break

        print('k: ', k)
        
        kmeans = KMeans(n_clusters=k).fit(driver_res)
        centroids = kmeans.cluster_centers_
        plt.scatter(driver_res['grid'], driver_res['positionOrder'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, marker="^")
        plt.xlabel("Grid")
        plt.ylabel("Finishing position")
        plt.savefig('cache/'+savename+'.png', dpi = 300)