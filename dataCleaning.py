import pandas as pd
import numpy as np


class DataCleaner():
    ##----------------
    #Series of Functions to clean the dataset
    ##----------------
    #Useful for getting the list of the columns with NaN cells
    #which can be used to get the entire values in those columns
    def ifCellIsEmptyGetColumn(self,FeaturesToCheck,dataFrame):
        pandaSeries = self.checkHowManyCellsEmpty(FeaturesToCheck,dataFrame)
        emptyColumns = pandaSeries.index[pandaSeries[FeaturesToCheck]!=0].tolist()
        return emptyColumns

    #Useful to see which columns have empty cells
    #and the number of empty cells
    def checkHowManyCellsEmpty(self,FeaturesToCheck,dataFrame):
        return dataFrame[FeaturesToCheck].isna().sum()

    #Useful when determining the shape of the 
    #dataFrame when the empty cells are dropped
    def shapeIfNaNDropped(self,dataFrame):
        return dataFrame.dropna().shape

    #Useful to determine the overall shape of the table (rows,columns)
    def shapeOfTable(self,dataFrame):
        return dataFrame.shape

    #useful for getting which rows are empty
    def whichRowsAreEmpty(self,FeaturesToCheck,dataFrame):
        dictEmptyRowsPerFeature = {}
        for feature in FeaturesToCheck:
            emptyRows = dataFrame.index[dataFrame[feature].isna()].tolist()
            dictEmptyRowsPerFeature[feature] = emptyRows
        return dictEmptyRowsPerFeature

    #Finding negative values where negative doesn't make sense
    def isThereNegative(self,FeaturesToCheck,dataFrame):
        dictNegPerFeature = {}
        for feature in FeaturesToCheck:
            rowWithNegative = dataFrame.index[(dataFrame[feature] < 0)].tolist()
            dictNegPerFeature[feature] = rowWithNegative
        return dictNegPerFeature 
    
    #Useful to see which columns have negative cells
    #and the number of negative cells
    def checkHowManyCellsNegative(self,FeatureToCheck, dataFrame):
        return (dataFrame[FeatureToCheck]<0).sum()

    #Useful for getting the list of the columns with negative cells
    #which can be used to get the entire values in those columns
    def ifCellIsNegativeGetColumn(self,FeaturesToCheck,dataFrame):
        pandaSeries = self.checkHowManyCellsNegative(FeaturesToCheck,dataFrame)
        emptyColumns = pandaSeries.index[pandaSeries[FeaturesToCheck]!=0].tolist()
        return emptyColumns
    #Useful for setting the negative cells which 
    # aren't supposed to be negative to absolute
    def setNegativeCellsAbs(self,FeaturesToSet,dataFrame):
        dataFrame[FeaturesToSet] = dataFrame[FeaturesToSet].abs()
        
    #Getting difference from mean to find overly big values
    def getDiffFromMean(self,FeaturesToCheck,dataFrame):
        createdColumns = []
        for feature in FeaturesToCheck:
            newColumns = "diffFromMean"+"-"+feature
            dataFrame[createdColumns] = (dataFrame[feature] - dataFrame[feature].mean()).abs()
            createdColumns.append(newColumns)
        return createdColumns

    #Useful for when the columns have mixed types
    def checkTypeMismatch(self,FeaturesToCheck,dataFrame):
        dictOfNaN = {}
        for feature in FeaturesToCheck:
            objectType = dataFrame[feature].map(type).value_counts()
            dictOfNaN[feature] = objectType.index.tolist()
        return dictOfNaN

    #If a value's type mismatches with the numeric
    # type in the column, it will be set to a numeric type
    def setToNumeric(self,FeaturesToNumeric,dataFrame):
        for feature in FeaturesToNumeric:
            print(feature)
            dataFrame[feature] = pd.to_numeric(dataFrame[feature])
            
    def isThereDuplicateRows(self,dataFrame):
        return dataFrame.duplicated().sum()

    #Useful for checking if some unique instance recur 
    # but with different spaces, spelling etc.
    def getUniqueInstancesPerCol(self,FeatureToCheck,dataFrame):
        return dataFrame[FeatureToCheck].unique()

    #useful for cleaning the unique instances to see if 
    # they recur but with different spaces, spelling etc.
    def getUniqueRemoveSpaceAndLowerCase(self,FeatureToCheck, dataFrame):
        arr = self.getUniqueInstancesPerCol(FeatureToCheck,dataFrame)
        arr = arr.astype(str)
        arr = np.char.lower(arr)
        arr = np.char.replace(arr," ","")
        return arr

    #Useful for checking if each data repeats anywhere in the array
    def checkIfDataRepeatsWithinArray(self,npArr):
        dictDataRepeats = {}
        values, counts = np.unique(npArr,return_counts=True)
        return values, counts

    #Useful for comparing a cleaned and uncleaned version of 
    #unique instances of categorical values so that we can
    #see if there is inconsistencies with the naming of a 
    #categorical value
    def compareCatValueModAndUnmod(self,valMod,countsMod,valUnmod,countsUnmod):
        valueMod = valMod.tolist()
        valueUnmod = valUnmod.tolist()
        compareUnmodAndMod = countsMod * countsUnmod
        countsMoreThanOne = np.where(compareUnmodAndMod>1)[0]
        counts = countsMoreThanOne.tolist()
        valueModXvalueUnmod = []
        for count in counts:
            valueModXvalueUnmod.append(valueMod[count]+" X "+valueUnmod[count])
        return valueModXvalueUnmod

    #Useful for checking the inconsistencies 
    #in value structure across a list of features
    def checkInconFromListOfCatValue(self,FeaturesToCheck, dataFrame):
        dictComparedForIncon = {}
        for feature in FeaturesToCheck:
            modArr = self.getUniqueInstancesPerCol(feature,dataFrame)
            unmodArr = self.getUniqueRemoveSpaceAndLowerCase(feature,dataFrame)
            modVal, modCount = self.checkIfDataRepeatsWithinArray(modArr)
            unmodVal, unmodCount = self.checkIfDataRepeatsWithinArray(unmodArr)
            compared = self.compareCatValueModAndUnmod(modVal,modCount,unmodVal,unmodCount)
            dictComparedForIncon[feature] = compared
        return dictComparedForIncon

    #Useful for when there are values that must 
    #stay within a certain range like lat and lon
    def valuesOutOfSetRange(self,FeatureToCheck,range,dataFrame):
        return dataFrame[(dataFrame[FeatureToCheck]<range[0])|(dataFrame[FeatureToCheck]>range[1])]

    #set date string type to panda date dtype
    def setDateToDtypeDate(self,dateColumn,dataFrame,format):
        dataFrame[dateColumn] = pd.to_datetime(dataFrame[dateColumn], format=format)

    #set a column of day of the week taken from the date    
    def setDayOfTheWeek(self,dateColumn,dataFrame):
        dataFrame["day_of_week"] = dataFrame[dateColumn].dt.dayofweek

    #Get pickup and delivery values where they are equal (that shouldn't be)
    def getPickupEqualsDelivery(self,pickup,delivery,dataFrame):
        return dataFrame[dataFrame[pickup]==dataFrame[delivery]]

    #If it turns out that pickup is equals to 
    #delivery (which is weird in itself)
    #we check if there is distance 0 
    #which strengthens the weirdness
    def checkOfAnyWithDistanceZero(self,distanceFeat,dataFrame):
        dataFrameForDistance = dataFrame[distanceFeat]
        return dataFrameForDistance[dataFrameForDistance==0]
    
    def isThereInfinity(self,FeaturesToCheck,dataFrame):
        return np.isinf(dataFrame[FeaturesToCheck]).sum()

    def dropNaNRows(self,dataFrame):
        return dataFrame.dropna()


   

