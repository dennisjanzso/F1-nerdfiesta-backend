import data_manager
import visualizer
import ml

class DataInterface():
    def __init__(self) -> None:
        self.DM = data_manager.DataManager()
        self.visualizer = visualizer.Visualizer(self.DM, None)
        self.predictor = ml.Predictor(self.DM, build_n=1)

    def getRacePlot(self, ticket, raceId):
        self.visualizer.strategy = visualizer.RacePlotter()
        self.visualizer.execute_strategy(raceId, save_name=ticket)

    def getDriverCluster(self, ticket, driverId):
        self.visualizer.strategy = visualizer.KmeansPlotter()
        self.visualizer.execute_strategy(driverId, savename=ticket)

    def getPastRaces(self):
        return self.DM.getPastRaces()

    def getFutureRaces(self):
        return self.DM.getFutureRaces(form='json')

    def getRacePrediction(self, ticket, raceId, weather):
        return self.predictor.predict_race(raceId, weather, ticket)

    def getRaceResults(self, raceId):
        return self.DM.getRaceResults(raceId, kind='json')

    def getRaceDetails(self, raceId):
        return self.DM.getRaceDetails(raceId, kind='json')

    def getCurrentDrivers(self):
        return {'drivers': list(self.DM.getSeasonDrivers(2021).T.to_dict().values())}

    def getDriverDetails(self, driverId):
        return {'driver': list(self.DM.getDriverDetails(driverId).T.to_dict().values())}