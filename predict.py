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
    "posted_rate", "distance","weight",
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

decemberChartMissing = [
    'pickup_lat', 'pickup_lon', 'delivery_lat', 
    'delivery_lon', 'market_index', 'quote_signal'
]

decemberChartInputs = [
    "pickup", "delivery", "distance",
    "equipment", "weight", "date", 
    "predicted_rate"
]

def saveCSV(dataFrame,savePath,newColumn,predictions = []):
    if len(predictions)!=0:
        dataFrame[newColumn] = predictions
    dataFrame.to_csv(savePath,index=False)
    
def predict(dataFrame,modelPath):    
    X_val = dataFrame[inputFeatures]
    # y_val = dataFrame[outputFeatures]
    model = CatBoostRegressor()
    model.load_model(modelPath)
    predictions = model.predict(X_val)
    # rmse = mean_squared_error(y_val, predictions) ** 0.5
    # print(f"RMSE: {rmse:.2f}")
    # baseline_predictions = [y_val.mean()] * len(y_val)
    # baseline_rmse = mean_squared_error(y_val,baseline_predictions) ** 0.5
    # print(f"Baseline RMSE: {baseline_rmse:.2f}")
    return predictions

if __name__ == "__main__":
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    # df = pd.read_csv("TemplatesCSV/train-test.csv")
    df = pd.read_csv("TemplatesCSV/validation.csv")
    dfValToSave = pd.read_csv("TemplatesCSV/validation.csv")
    dfValPredToSave = pd.read_csv("TemplatesCSV/validation-predictions-template.csv")
    dfDecemberChart = pd.read_csv("TemplatesCSV/december-chart-inputs.csv")
    for decemberMissing in decemberChartMissing:
        dfDecemberChart[decemberMissing] = np.nan
    # print(cleaner.ifCellIsEmptyGetColumn(columns[:13],df))
    # print(cleaner.checkHowManyCellsEmpty(columns[:13],df))
    # print(cleaner.checkHowManyCellsNegative(numericalFeatures[1:5],df))
    # negCol = cleaner.ifCellIsNegativeGetColumn(numericalFeatures[1:5],df)
    # newSer = df[negCol]
    # print(newSer[(newSer[negCol]<0)].dropna())
    # print(cleaner.isThereInfinity(numericalFeatures[1:],df))
    # print(cleaner.checkTypeMismatch(columns[:13],df))
    # print(cleaner.isThereDuplicateRows(df))
    # print(cleaner.checkInconFromListOfCatValue(categoricalFeat[:3],df))
    # print(cleaner.getPickupEqualsDelivery("pickup","delivery",df))
    # print(cleaner.checkOfAnyWithDistanceZero("distance",df))
    
    listNegColumns = cleaner.ifCellIsNegativeGetColumn(numericalFeatures[1:5],df)
    cleaner.setNegativeCellsAbs(listNegColumns,df)
    cleaner.setDateToDtypeDate("date",df,"%Y-%m-%d")
    cleaner.setDayOfTheWeek("date",df)
    cleaner.setDateToDtypeDate("date",dfDecemberChart,"%Y-%m-%d")
    cleaner.setDayOfTheWeek("date",dfDecemberChart)
    predictions = predict(df,"model6.cbm")
    decPred = predict(dfDecemberChart,"model6.cbm")
    dfDecemberChart = dfDecemberChart[decemberChartInputs].copy()
    saveCSV(dfValToSave,"validation.csv","predicted_rate",predictions)
    saveCSV(dfValPredToSave,"validation-predictions.csv","predicted_rate",predictions)
    saveCSV(dfDecemberChart,"december-chart-inputs.csv","predicted_rate",decPred)
    
    