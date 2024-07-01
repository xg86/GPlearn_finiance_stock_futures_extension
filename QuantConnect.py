# region imports
from AlgorithmImports import *
from gplearn.genetic import SymbolicRegressor, SymbolicTransformer
import joblib


# endregion

class GPlearnExampleAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2022, 7, 4)
        self.SetEndDate(2022, 7, 8)
        self.SetCash(100000)
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        training_length = 252 * 2
        self.training_data = RollingWindow[float](training_length)
        history = self.History[TradeBar](self.symbol, training_length, Resolution.Daily)
        for trade_bar in history:
            self.training_data.Add(trade_bar.Close)

        transformer_model_key = "Transformer"
        regressor_model_key = "Regressor"
        if self.ObjectStore.ContainsKey(transformer_model_key) and self.ObjectStore.ContainsKey(regressor_model_key):
            transformer_file_name = self.ObjectStore.GetFilePath(transformer_model_key)
            regressor_file_name = self.ObjectStore.GetFilePath(regressor_model_key)
            self.gp_transformer = joblib.load(transformer_file_name)
            self.model = joblib.load(regressor_model_key)

        else:
            function_set = ['add', 'sub', 'mul', 'div',
                            'sqrt', 'log', 'abs', 'neg', 'inv',
                            'max', 'min']
            self.gp_transformer = SymbolicTransformer(function_set=function_set)
            self.model = SymbolicRegressor()
            self.Train(self.my_training_method)

        self.Train(self.DateRules.Every(DayOfWeek.Sunday), self.TimeRules.At(8, 0), self.my_training_method)

    def get_features_and_labels(self, n_steps=5):
        training_df = list(self.training_data)[::-1]
        daily_pct_change = ((np.roll(training_df, -1) - training_df) / training_df)[:-1]

        features = []
        labels = []
        for i in range(len(daily_pct_change) - n_steps):
            features.append(daily_pct_change[i:i + n_steps])
            labels.append(daily_pct_change[i + n_steps])
        features = np.array(features)
        labels = np.array(labels)

        return features, labels

    def my_training_method(self):
        features, labels = self.get_features_and_labels()

        # Feature engineering
        self.gp_transformer.fit(features, labels)
        gp_features = self.gp_transformer.transform(features)
        new_features = np.hstack((features, gp_features))

        # Fit the regression model with transformed and raw features.
        self.model.fit(new_features, labels)

    def OnData(self, slice):
        features, _ = self.get_features_and_labels()

        # Get transformed features
        gp_features = self.gp_transformer.transform(features)
        new_features = np.hstack((features, gp_features))

        # Get next prediction
        prediction = self.model.predict(new_features)
        prediction = float(prediction.flatten()[-1])

        if prediction > 0:
            self.SetHoldings(self.symbol, 1)
        elif prediction < 0:
            self.SetHoldings(self.symbol, -1)

    def OnEndOfAlgorithm(self):
        transformer_model_key = "Transformer"
        regressor_model_key = "Regressor"
        transformer_file_name = self.ObjectStore.GetFilePath(transformer_model_key)
        regressor_file_name = self.ObjectStore.GetFilePath(regressor_model_key)
        joblib.dump(self.gp_transformer, transformer_file_name)
        joblib.dump(self.model, regressor_file_name)
        self.ObjectStore.Save(transformer_model_key)
        self.ObjectStore.Save(regressor_model_key)