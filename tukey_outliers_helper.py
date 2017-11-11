import numpy as np


class TukeyOutliersHelper(object):
    default_k = 1.5

    def getBoundsFromDataFrame(self, df, kk, outlier_columns):
        types = df[outlier_columns].dtypes
        for col in outlier_columns:
            cur_data_col = df[col]
            types[col] = self.getMedianBoundaries(cur_data_col, kk=kk)
        return types.copy()

    def removeOutliers(self, data, bounds):
        """returns the data with the removed outliers and the indices that survived"""

        fatalRowInds = self.getOutlierDataPoints(data, bounds)
        survivors = list(set(data.index).difference(fatalRowInds))
        assert len(data) == (len(survivors) + len(fatalRowInds))
        return data.loc[survivors], data.loc[fatalRowInds]

    @staticmethod
    def getOutliersIndices(data, bounds, filtering=lambda arr: np.repeat(True, len(arr))):
        """bounds are a series of boundaries and data is a dataset matrix"""
        outliers_inds = bounds.copy()

        for col in bounds.keys():
            curBounds = bounds[col]
            curCol = data[col][filtering]

            smaller_args = np.argwhere(curCol < curBounds[0]).flatten()
            bigger_args = np.argwhere(curBounds[1] < curCol).flatten()

            outliers_inds[col] = np.hstack((smaller_args, bigger_args))

        return outliers_inds

    @staticmethod
    def countOutliersDataPoints(data, bounds, filtering=lambda arr: np.repeat(True, len(arr))):
        """bounds are a series of boundaries and data is a dataset matrix"""
        counts = bounds.copy()

        for col in bounds.keys():
            curBounds = bounds[col]
            curCol = data[col][filtering]

            smaller_args = np.argwhere(curCol < curBounds[0])
            bigger_args = np.argwhere(curCol > curBounds[1]) #curBounds[1] < curCol) <-- this did not work with all versions

            counts[col] = len(smaller_args) + len(bigger_args)

        return counts

    @staticmethod
    def getOutlierDataPoints(data, bounds):
        """bounds are a series of boundaries and data is a dataset matrix"""
        curDict = bounds.to_dict()

        fatalRowInds = set()

        for curColName in curDict:
            curBounds = curDict[curColName]
            curCol = data[curColName].to_frame()
            # print curCol
            # exit(1)

            for rowInd in data.index:
                curValue = curCol.loc[rowInd].values[0]
                if (curValue < curBounds[0]) or (curBounds[1] < curValue):  # outlier detected
                    fatalRowInds.add(rowInd)

        return fatalRowInds

    @staticmethod
    def getOutlierDataPointsNumpy(data, bounds):
        """data is a numpy matrix (rows are the instances, columns are the attributes) and
            bounds is a list of tuples that contain the boundaries"""

        fatalRowInds = set()

        dataLen = len(data)

        for colIndex in range(data.shape[1]):
            curBounds = bounds[colIndex]
            curCol = data[:, colIndex]

            for rowInd in range(dataLen):
                curValue = curCol[rowInd]
                isOutlier = (curValue < curBounds[0]) or (curBounds[1] < curValue)
                if isOutlier:
                    fatalRowInds.add(rowInd)

        return fatalRowInds

    def getLooseBoundaries(self, col, k=default_k):
        lowboundMedian, highboundMedian = self.getBoundaries(col, k=k, median=True)
        lowboundMean, highboundMean = self.getBoundaries(col, k=k, median=False)
        return min(lowboundMedian, lowboundMean), max(highboundMedian, highboundMean)

    @staticmethod
    def getMedianBoundaries(col, kk=default_k):
        """alternative use k=3 for data that are far out"""
        q1, q3 = np.percentile(col, [25, 75])

        # q1 - k(q3 - q1), q3 + k(q3-q1)
        lowbound, highbound = q1 - kk * (q3 - q1), q3 + kk * (q3 - q1)

        # q1, q3
        return lowbound, highbound

    @staticmethod
    def getBoundaries(col, k=default_k, median=True):
        """alternative use k=3 for data that are far out
        the theory says to work with medians but this does not work always"""

        sortedList = col.sort_values(ascending=True).as_matrix()

        lenList = len(sortedList)

        if lenList % 2 == 1:
            ind = int(lenList / 2)
            mylist = np.delete(sortedList, ind)  # plus one is not necessary
        else:
            mylist = sortedList

        halfway = int(len(mylist) / 2)
        lowerhalf = mylist[:halfway]
        upperhalf = mylist[halfway:]

        q1 = np.median(lowerhalf) if median else np.mean(lowerhalf)
        q3 = np.median(upperhalf) if median else np.mean(upperhalf)

        # q1 - k(q3 - q1), q3 + k(q3-q1)
        lowbound, highbound = q1 - k * (q3 - q1), q3 + k * (q3 - q1)

        # q1, q3
        return lowbound, highbound
