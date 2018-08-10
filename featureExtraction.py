import pandas as pd
from sklearn import base
import datetime
import re
import numpy as np

def getDateObject(inputString):
    return datetime.datetime.strptime(inputString,"%Y-%m-%d %H:%M:%S")


class ColumnSelector(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self,columnName):
        self.columnName = columnName
        
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        usersDF, tweetsDF = X
        
        allVecs=[]
        for idx,row in usersDF.iterrows():
            allVecs.append(row[self.columnName])
        return np.array(allVecs).reshape(len(allVecs),-1)


class NumberOfDaysBetweenCreationAndLastTweet(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self):
        ''''''
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        usersDF, tweetsDF = X
        
        allVecs=[]
        for idx,row in usersDF.iterrows():
            userID = row['UserID']
            userTweets = tweetsDF[tweetsDF['UserID']==userID]
            if(len(userTweets)==0):
                allVecs.append(0)
            else:
                lastDate = getDateObject(max(userTweets['CreatedAt']))
                firstDate = getDateObject(row.CreatedAt)
                allVecs.append((lastDate-firstDate).days)
        return np.array(allVecs).reshape(len(allVecs),-1)


class NumberOfDaysBetweenFirstAndLastTweetCollected(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self):
        ''''''
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        usersDF, tweetsDF = X
        
        allVecs=[]
        for idx,row in usersDF.iterrows():
            userID = row['UserID']
            userTweets = tweetsDF[tweetsDF['UserID']==userID]
            if(len(userTweets)==0):
                allVecs.append(0)
            else:
                lastDate = getDateObject(max(userTweets['CreatedAt']))
                firstDate = getDateObject(min(userTweets['CreatedAt']))
                diff = (lastDate-firstDate)
                allVecs.append(diff.days+diff.seconds/60/60/24)
        return np.array(allVecs).reshape(len(allVecs),-1)

class NumberOfDaysBetweenCreationAndFirstTweet(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self):
        ''''''
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        usersDF, tweetsDF = X
        
        allVecs=[]
        for idx,row in usersDF.iterrows():
            userID = row['UserID']
            userTweets = tweetsDF[tweetsDF['UserID']==userID]
            
            creationDate = getDateObject(row['CreatedAt'])

            
            if(len(userTweets)==0):
                allVecs.append(-1)
            else:
                firstTweetDate = getDateObject(min(userTweets['CreatedAt']))
                creationDate = getDateObject(row['CreatedAt'])
                firstTweetDate = getDateObject(min(userTweets['CreatedAt']))
                allVecs.append((firstTweetDate-creationDate).days)

        return np.array(allVecs).reshape(len(allVecs),-1)

class NumberOfTweets(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self):
        ''''''
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        usersDF, tweetsDF = X
        
        allVecs=[]
        for idx,row in usersDF.iterrows():
            userID = row['UserID']
            userTweets = tweetsDF[tweetsDF['UserID']==userID]
            allVecs.append(len(userTweets))

        return np.array(allVecs).reshape(len(allVecs),-1)


class DivideColumns(base.BaseEstimator, base.TransformerMixin):
    def __init__(self,numerator,denominator):
        self.numerator = numerator
        self.denominator = denominator
        ''''''
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        
        numerator = self.numerator.fit_transform(X)
        denominator = self.denominator.fit_transform(X)
      
        with np.errstate(divide='ignore', invalid='ignore'):
            #If numerator is zero, then replace numerator 
            numerator[denominator == 0] = 0            
            denominator[denominator == 0] = -1            

            return (np.divide(numerator,denominator))


class AverageNumberOfCharactersPerTweet(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self):
        ''''''
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        usersDF, tweetsDF = X
        
        allVecs=[]
        for idx,row in usersDF.iterrows():
            userID = row['UserID']
            userTweets = tweetsDF[tweetsDF['UserID']==userID]
            
            if(len(userTweets)==0):
                allVecs.append(-1)
            else:
                allVecs.append(userTweets.Tweet.apply(lambda x: len(str(x))).mean())

        return np.array(allVecs).reshape(len(allVecs),-1)


class AverageNumberOfWordsPerTweet(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self):
        ''''''
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        usersDF, tweetsDF = X
        
        allVecs=[]
        for idx,row in usersDF.iterrows():
            userID = row['UserID']
            userTweets = tweetsDF[tweetsDF['UserID']==userID]
            
            if(len(userTweets)==0):
                allVecs.append(-1)
            else:
                allVecs.append(userTweets.Tweet.apply( lambda x : len(str(x).split())).mean())

        return np.array(allVecs).reshape(len(allVecs),-1)


from zlib import compress



class ZipCompressionRatio(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self):
        ''''''
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        usersDF, tweetsDF = X
        
        allVecs=[]
        for idx,row in usersDF.iterrows():
            userID = row['UserID']
            userTweets = tweetsDF[tweetsDF['UserID']==userID]
            
            if(len(userTweets)==0):
                allVecs.append(-1)
            else:
                bytesString = str.encode("".join(str(userTweets.Tweet)))
                                                 
                allVecs.append(len(bytesString)/len(compress(bytesString,6)))

        return np.array(allVecs).reshape(len(allVecs),-1)


class NumberOfMentions(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self):
        ''''''
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        usersDF, tweetsDF = X
        
        allVecs=[]
        for idx,row in usersDF.iterrows():
            userID = row['UserID']
            userTweets = tweetsDF[tweetsDF['UserID']==userID]
            
            if(len(userTweets)==0):
                allVecs.append(-1)
            else:
                allVecs.append(userTweets.Tweet.apply( lambda x : len(re.findall(r"(@\w+)",str(x)))).sum())

        return np.array(allVecs).reshape(len(allVecs),-1)


class AverageNumberOfMentionsPerTweet(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self):
        ''''''
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        usersDF, tweetsDF = X
        
        allVecs=[]
        for idx,row in usersDF.iterrows():
            userID = row['UserID']
            userTweets = tweetsDF[tweetsDF['UserID']==userID]
            
            if(len(userTweets)==0):
                allVecs.append(-1)
            else:
                allVecs.append(userTweets.Tweet.apply( lambda x : len(re.findall(r"(@\w+)",str(x)))).mean())

        return np.array(allVecs).reshape(len(allVecs),-1)


class NumberOfLinks(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self):
        ''''''
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        usersDF, tweetsDF = X
        
        allVecs=[]
        for idx,row in usersDF.iterrows():
            userID = row['UserID']
            userTweets = tweetsDF[tweetsDF['UserID']==userID]
            
            if(len(userTweets)==0):
                allVecs.append(-1)
            else:
                allVecs.append(userTweets.Tweet.apply( lambda x : len(re.findall(r"(http[s]?://)",str(x)))).sum())

        return np.array(allVecs).reshape(len(allVecs),-1)


class AverageDaysBetweenTweets(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self):
        ''''''
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        usersDF, tweetsDF = X
        
        allVecs=[]
        for idx,row in usersDF.iterrows():
            userID = row['UserID']
            userTweets = tweetsDF[tweetsDF['UserID']==userID]
            if(len(userTweets)==0):
                allVecs.append(0)
            else:
                lastDate = getDateObject(max(userTweets['CreatedAt']))
                firstDate = getDateObject(min(userTweets['CreatedAt']))
                diff = (lastDate-firstDate)
                highResDays = diff.days+diff.seconds/60/60/24
                allVecs.append(highResDays/len(userTweets))
        return np.array(allVecs).reshape(len(allVecs),-1)

class ApplyFunction(base.BaseEstimator, base.TransformerMixin):
    def __init__(self,estimator,function):
        self.estimator = estimator
        self.function = function
    def fit(self, X, y=None):
        return self
    def transform(self, X):       
        output = self.estimator.fit_transform(X)+1
        return self.function(output)



