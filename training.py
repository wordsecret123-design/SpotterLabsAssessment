import numpy as np
import pandas as pd
from dataCleaning import DataCleaner as dc
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

cleaner = dc()
columns = [
    "load_id","pickup","delivery","pickup_lat","pickup_lon",
    "delivery_lat","delivery_lon","distance","equipment","weight",	
    "date",	"market_index",	"quote_signal","posted_rate"
]
numericalFeatures = [
    "distance","weight","posted_rate",
    "quote_signal", "market_index",
    "pickup_lat","pickup_lon",
    "delivery_lat","delivery_lon"
]

categoricalFeat = [
    "pickup","delivery",
    "equipment", "day_of_week"
]

coordinates = [
    "pickup_lat","pickup_lon",
    "delivery_lat","delivery_lon"
]

inputFeatures = [
    "pickup","delivery","pickup_lat","pickup_lon",
    "delivery_lat","delivery_lon","distance","equipment",
    "weight", "market_index", "quote_signal","day_of_week"
]

outputFeatures = [
    "posted_rate"
]


#by dropping the NaN this function should still work, 
#except rows with NaN's are no longer included
def trainingValidationSplit(featuresWithNaN,dataFrame,dateColumn):
    areNaN = dataFrame[dataFrame[featuresWithNaN].isna().any(axis=1)]
    areNotNaN = dataFrame[dataFrame[featuresWithNaN].notna().all(axis=1)]
    areNotNaN = areNotNaN.sort_values(dateColumn)
    split_index = int((len(areNotNaN)+len(areNaN))*0.2)
    split_index = int(len(areNotNaN) - split_index)
    trainSetDF = pd.concat([areNotNaN[:split_index],areNaN])
    validSetDF = areNotNaN[split_index:]
    return trainSetDF,validSetDF

def trainingValidationSplitWithNaN(dataFrame,dateColumn):
    dataFrame = dataFrame.sort_values(dateColumn)
    split_index = int(len(df)*0.8)
    trainSetDF = dataFrame[:split_index]
    validSetDF = dataFrame[split_index:]
    return trainSetDF, validSetDF

def train(dataFrame):    
    # depths = [6, 8, 10, 12, 14, 16]
    depths = [6]
    trainSetDF,validSetDF = trainingValidationSplitWithNaN(dataFrame,"date")
    X_train = trainSetDF[inputFeatures]
    y_train = trainSetDF[outputFeatures]
    X_val = validSetDF[inputFeatures]
    y_val = validSetDF[outputFeatures]
    for depth in depths:
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=depth,
            loss_function="RMSE",
            verbose=1,
            early_stopping_rounds=500
        )
        model.fit(
            X_train,
            y_train,
            cat_features=categoricalFeat,
            eval_set=(X_val, y_val)
        )
        predictions = model.predict(X_val)
        rmse = mean_squared_error(y_val, predictions) ** 0.5
        print(f"Depth: {depth}, RMSE: {rmse:.2f}")
        model.save_model("model"+str(depth)+".cbm")
        baseline_predictions = [y_train.mean()] * len(y_val)
        baseline_rmse = mean_squared_error(y_val,baseline_predictions) ** 0.5
        print(f"Baseline RMSE: {baseline_rmse:.2f}")

if __name__ == "__main__":
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    df = pd.read_csv("train-test.csv")
    # featuresWithNaN = cleaner.ifCellIsEmptyGetColumn(columns,df)
    listNegColumns = cleaner.ifCellIsNegativeGetColumn(numericalFeatures[:5],df)
    cleaner.setNegativeCellsAbs(listNegColumns,df)
    cleaner.setDateToDtypeDate("date",df,"%Y-%m-%d")
    cleaner.setDayOfTheWeek("date",df)
    train(df)
    # trainingValidationSplit(featuresWithNaN,df,"date")
    