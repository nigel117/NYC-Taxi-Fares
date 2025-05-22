from math import sqrt
import seaborn as sns
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import sys

def main():
    df, testData = callset()

    df['Distance'] = hav_dist(df['pickup_latitude'],df['pickup_longitude'],df['dropoff_latitude'],df['dropoff_longitude'],df)
    testData['Distance'] = hav_dist(df['pickup_latitude'],df['pickup_longitude'],df['dropoff_latitude'],testData['dropoff_longitude'],testData)

    df['Distance'],df['fare_amount'] = cleaning(df)
    testData['Distance'],testData['fare_amount'] = cleaning(testData)

    df = df.dropna()
    
    
    kmeans(df, testData)
    
 

    
    
    


def callset():
 
    df = pd.read_csv('train.csv', nrows = 13000)
    testData = df[10000:12489]
    
    df = df[0:10000]
    print(df.info)
    return df, testData


def hav_dist(lat1, lon1, lat2, lon2,df):
    
    distance = np.arccos(np.sin(np.radians(lat1))* np.sin(np.radians(lat2))+ np.cos(np.radians(lat1))* np.cos(np.radians(lat2))* np.cos(np.radians(lon2 - lon1)))* 6371

    print(distance)
   
    return distance
  

def calcShannonEnt(distanceRanges,fareRanges):

    numDistanceEntries = len(distanceRanges) 
    numFareEntries = len(fareRanges)

    labelCounts = {}
    fareCounts = {}
    for i in range(len(fareRanges)): 
        currentLabelf = fareRanges[i]
        
        if currentLabelf not in fareCounts.keys():
            fareCounts[currentLabelf] = 0
        
        fareCounts[currentLabelf] += 1

    print('FareCounts:', fareCounts)

    for i in range(len(distanceRanges)): 
        currentLabel = distanceRanges[i]
        
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        
        labelCounts[currentLabel] += 1

    print('labelCounts:', labelCounts)

    shannonEnt = 0.0
    shannonEnt2 = 0.0

    for key in labelCounts: 
        prob = float(labelCounts[key])/numDistanceEntries 
        shannonEnt = shannonEnt - prob * np.log2(prob) 
    print("Entropy for Distance: " + str(shannonEnt))

    for key in fareCounts: 
        prob = float(fareCounts[key])/numFareEntries
        shannonEnt2 = shannonEnt2 - prob * np.log2(prob) 
    print("Entropy for fare amounts: " +  str(shannonEnt2))

    return shannonEnt, shannonEnt2
    

def cleaning(df):
    print('old size: %d' % len(df))
    
    df = df.dropna()
    remove =  df['fare_amount'] < 1
    df = df[~remove]
    remove =  (df['Distance']) <= 0  
    df = df[~remove]

    remove = (df['fare_amount'] > 60)
    df = df[~remove]
    remove = (df['Distance'] > 25)
    df = df[~remove]

    remove = (df['fare_amount'] > 15) & (df['Distance'] < 1)
    df = df[~remove]
    

    remove = (df['fare_amount'] > 23) & (df['Distance'] <3)
    df = df[~remove]
    remove = (df['fare_amount'] > 38) & (df['Distance'] <10)
    df = df[~remove]
    remove = (df['fare_amount'] > 54) & (df['Distance'] <15)
    df = df[~remove]
    remove = (df['fare_amount'] > 58) & (df['Distance'] <20)
    df = df[~remove]
    remove = (df['fare_amount'] <8) & (df['Distance'] > 5)
    df = df[~remove]
    remove = (df['fare_amount'] <12) & (df['Distance'] > 8)
    df = df[~remove]
    remove = (df['fare_amount'] <20) & (df['Distance'] > 15)
    df = df[~remove]


    print('new size after removing nan values and outliers: %d' % len(df))
    '''
    dist = np.array(df['Distance'])
    fare = np.array(df['fare_amount'])
    plt.xlabel('Distances')
    plt.ylabel('Fare amount')
    plt.plot(dist, fare, 'o')
    plt.show()
    '''
    return df['Distance'],df['fare_amount']

def kmeans(df, testData):
    a = df['Distance']
    b = df['fare_amount']
    x = np.array(a)
    y = np.array(b)
 
    predict_a = testData['Distance']
    predict_b = testData['fare_amount']

    predict_x = np.array(predict_a)
    predict_y = np.array(predict_b)

    data = list(zip(x, y))
    predictData = list(zip(predict_x, predict_y))


    inertias = []

    for i in range(1,11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    '''
    plt.plot(range(1,11), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()
    '''

    kmeans = KMeans(n_clusters=4)


    kmeans.fit(data)

    #clusterIndex = kmeans.predict(predictData)

    #for i in clusterIndex:
        #print("   Predicted Cluster = " + str(i))

    plt.xlabel('Distance')
    plt.ylabel('Fares')
    plt.scatter(x, y, c=kmeans.labels_)
    plt.show()

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(data)
    clusters = kmeans.fit_predict(data)

    
    fareRanges = []
    testFareRanges = []
    
    for cluster_id in range(4):
        cluster_data = df[clusters== cluster_id]

        max_fare_amount = cluster_data['fare_amount'].max()
        min_fare_amount = cluster_data['fare_amount'].min()

        max_distance = cluster_data['Distance'].max()
        min_distance = cluster_data['Distance'].min()

        print(f"Cluster {cluster_id + 1}:")
        print("Max Fare Amount:", max_fare_amount)
        print("Min Fare Amount:", min_fare_amount)
        print("Max Distance:", max_distance)
        print("Min Distance:", min_distance)

    
    for d in df['Distance']: 
        if d >= 0.0000949 and d <= 5.263257678375247:
            fareRanges.append("$2.5-$11.0")
        elif d >= 0.0698056605623137 and d <= 10.034921839127298:
            fareRanges.append("$9.3-$21.54")
        elif d >= 2.6611818397438975 and d <= 19.424791774793267:
            fareRanges.append("$19.7-$40.04")
        elif d >= 10.03487959031889 and d <= 24.690884111310794:
            fareRanges.append("$38.1-$58.0")

    for d in testData['Distance']: 
        if d >= 0.0000949 and d <= 5.263257678375247:
            testFareRanges.append("$2.5-$11.0")
        elif d >= 0.0698056605623137 and d <= 10.034921839127298:
            testFareRanges.append("$9.3-$21.54")
        elif d >= 2.6611818397438975 and d <= 19.424791774793267:
            testFareRanges.append("$19.7-$40.04")
        elif d >= 10.03487959031889 and d <= 24.690884111310794:
            testFareRanges.append("$38.1-$58.0")

    features = ['Distance']
    d = {'$2.5-$11.0': 1, '$9.3-$21.54': 2, '$19.7-$40.04': 3, '$38.1-$58.0': 4}
    df['fare_ranges'] = fareRanges
  
    
    X1 = df[features]
    Y1 = df['fare_ranges'].map(d)

    clf = DecisionTreeClassifier()
    clf = clf.fit(X1, Y1)

   
    tree.plot_tree(clf, feature_names=features)
    plt.show()

 
    print(f"Distance of 10 fare range prediction: {clf.predict([[10]])}")
    print("[1] means $2.5-$11.0")
    print("[2] means $9.3-$21.54")
    print("[3] means $19.7-$40.04")
    print("[4] means $38.1-$58.0")
    print(f"Distance of 20 fare range prediction: {clf.predict([[20]])}")
    print("[1] means $2.5-$11.0")
    print("[2] means $9.3-$21.54")
    print("[3] means $19.7-$40.04")
    print("[4] means $38.1-$58.0")
    print(f"Distance of 1 fare range prediction: {clf.predict([[1]])}")
    print("[1] means $2.5-$11.0")
    print("[2] means $9.3-$21.54")
    print("[3] means $19.7-$40.04")
    print("[4] means $38.1-$58.0")
    




if __name__ == "__main__":
    main()
