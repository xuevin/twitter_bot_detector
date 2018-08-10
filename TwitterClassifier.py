import pickle
from sklearn.pipeline import FeatureUnion
import tweepy
import simplejson as json
from featureExtraction import *
import datetime
import pandas as pd
from sklearn import pipeline



with open("/home/vagrant/datacourse/capstone/twitter_secrets.json.nogit") as fh:
    secrets = json.loads(fh.read())



auth = tweepy.OAuthHandler(secrets['api_key'], secrets['api_secret'])
auth.set_access_token(secrets['access_token'], secrets['access_token_secret'])
api = tweepy.API(auth)



usersHeader = ['UserID',
'CreatedAt',
'CollectedAt',
'NumerOfFollowings',
'NumberOfFollowers',
'NumberOfTweets',
'LengthOfScreenName',
'LengthOfDescriptionInUserProfile']



tweetsHeader = ['UserID','TweetID','Tweet','CreatedAt']



def getUsersFromListOfTweets(thisList):
    
    a = []
    b = []
    c = []
    d = []
    e = []
    f = []
    g = []
    
    for i in thisList:

        a.append(i.user.screen_name)
        b.append(int(i.user.id))
        c.append(str(i.user.created_at))
        d.append(int(i.user.friends_count))
        e.append(int(i.user.followers_count))
        f.append(int(i.user.statuses_count))
        g.append(len(i.user.description))
    
    return pd.DataFrame({'UserID':b,
                        'CreatedAt':c,
                        'CollectedAt':datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'NumerOfFollowings':d,
                        'NumberOfFollowers':e,
                        'NumberOfTweets':f,
                        'LengthOfScreenName':[len(i) for i in a],
                        'LengthOfDescriptionInUserProfile':g,
                        'ScreenName':a})



def getLast200TweetsFromUserID(userID):
    user = api.get_user(user_id = str(userID))
    userTweet = user.timeline(count=200)
    

    
    
    a = []
    b = []
    c = []
    
    for i in userTweet:
        #print(i.id, i.text, i.created_at)

        a.append(int(i.id_str))
        b.append(i.text)
        c.append(str(i.created_at))

    return pd.DataFrame({'UserID':[userID]*len(a),
                'TweetID':a,
                'Tweet':b,
                'CreatedAt':c})
    
    
    


# # Write functions to get evaluate a user


union = FeatureUnion([
    ('Log10NumberOfTweetsTotal',                      ApplyFunction(ColumnSelector('NumberOfTweets'),np.log10)),#OKAY 
    ('Log10NumberOfFollowers',                        ApplyFunction(ColumnSelector('NumberOfFollowers'),np.log10)),#OKAY 
    ('Log10NumberOfFollowings',                       ApplyFunction(ColumnSelector('NumerOfFollowings'),np.log10)),#OKAY 
    ('RatioFollowerFollowings',                       DivideColumns(ColumnSelector('NumberOfFollowers'),ColumnSelector('NumerOfFollowings'))),#OKAY 
    ('AverageNumberOfWordsPerTweet',                  AverageNumberOfWordsPerTweet()),     #OKAY 
    ('AverageNumberOfCharactersPerTweet',             AverageNumberOfCharactersPerTweet()),#OKAY 
    ('AverageTweetsPerDay',                           DivideColumns(NumberOfTweets(),NumberOfDaysBetweenFirstAndLastTweetCollected())),#OKAY    
    ('NumberOfMentionsForCollectedTweets',            NumberOfMentions()),                #OKAY 
    ('AverageNumberOfMentionsPerDay',                 DivideColumns(NumberOfMentions(),NumberOfDaysBetweenFirstAndLastTweetCollected())), #OKAY
    ('AverageNumberOfMentionsPerTweet',               AverageNumberOfMentionsPerTweet()),
    ('NumberOfLinksForCollectedTweets',               NumberOfLinks()), #OKAY
    ('AverageNumberOfLinksPerDay',                    DivideColumns(NumberOfLinks(),NumberOfDaysBetweenFirstAndLastTweetCollected())), #OKAY
    ('ZipCompressionRatio',                           ZipCompressionRatio()),#OKAY
    ('AverageDaysBetweenTweets',                      AverageDaysBetweenTweets()), #OKAY
    ('NumberOfDaysBetweenCreationAndLastTweet',       NumberOfDaysBetweenCreationAndLastTweet()), #OKAY
    ('NumberOfDaysBetweenFirstAndLastTweetCollected', NumberOfDaysBetweenFirstAndLastTweetCollected()) #OKAY
    ])


# In[9]:


def printUserStats(screenName,encodedUser):
    print(screenName)
    for  i,j in zip(features,encodedUser.flatten()):
        print(str(i).ljust(50),j)


# In[10]:


def scoreUsers(dataFrameOfUsers):
    encoding_pipe = pipeline.Pipeline([
        ('unionFeatures', union)
    ])
    for i,row in dataFrameOfUsers.iterrows():
        last200Tweets = getLast200TweetsFromUserID(row["UserID"])
        encodedUser = encoding_pipe.fit_transform((pd.DataFrame([row]),last200Tweets))
        isReal = modelPipeline.predict(encodedUser)[0]
        probs = modelPipeline.predict_proba(encodedUser)
        return row['ScreenName'],isReal,probs,encodedUser


# In[11]:


def getUserDF(screenName,userId=None):
    
    a = []
    b = []
    c = []
    d = []
    e = []
    f = []
    g = []
    
    while(1):
        try:
            if(userId):
                user = api.get_user(user_id = str(userId))
            else:
                user = api.get_user(screen_name = str(screenName))
            break
            
        except tweepy.RateLimitError:
            #print("Rate Limit: %s" % str(thisID))
            time.sleep(60 * 15)
            continue
        except tweepy.TweepError as e:
            #print(e.args[0])
            return pd.DataFrame({'UserID':b,
                    'CreatedAt':c,
                    'CollectedAt':datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'NumerOfFollowings':d,
                    'NumberOfFollowers':e,
                    'NumberOfTweets':f,
                    'LengthOfScreenName':[len(i) for i in a],
                    'LengthOfDescriptionInUserProfile':g,
                    'ScreenName':a})
    
    

    

    

    a.append(user.screen_name)
    b.append(int(user.id))
    c.append(str(user.created_at))
    d.append(int(user.friends_count))
    e.append(int(user.followers_count))
    f.append(int(user.statuses_count))
    g.append(len(user.description))
    
    return pd.DataFrame({'UserID':b,
                    'CreatedAt':c,
                    'CollectedAt':datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'NumerOfFollowings':d,
                    'NumberOfFollowers':e,
                    'NumberOfTweets':f,
                    'LengthOfScreenName':[len(i) for i in a],
                    'LengthOfDescriptionInUserProfile':g,
                    'ScreenName':a})

modelPipeline = pickle.load(open("/home/vagrant/datacourse/capstone/twitter.clfPipelne",'rb'))

#SN,prediction,probs, encodedUser= scoreUsers(getUserDF("xuevin"))
#print(SN,prediction,probs)


