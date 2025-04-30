import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#reading in data 
df = pd.read_csv('maddennfl24fullplayerratings.csv')
#print(df)
#creating df with just QBs
qbs_df = df[df['Position'] == 'QB']
qbs_df.info()

num_feature = ['Speed', 'Acceleration', 'Strength', 'Agility', 'Awareness', 'Carrying', 'Throw Power', 'Break Tackle', 'Injury','Stamina', 'Toughness', 'Break Sack', 'Throw Under Pressure', 'Throw Accuracy Short', 'Throw Accuracy Mid', 'Throw Accuracy Deep', 'Play Action' ]
qb_selected = qbs_df[num_feature]
#print(qb_selected)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(qb_selected)


interia_scores = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    interia_scores.append(kmeans.inertia_)
    

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), interia_scores, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('interia_scores')
plt.savefig('elbow_plot.png')
plt.show()


kmeans = KMeans(n_clusters=3, random_state=50)
clusters = kmeans.fit_predict(scaled_data)
qbs_df['Cluster'] = clusters


for cluster_num in range(3):
    print(f"\nCluster {cluster_num} example players:")
    print(qbs_df[qbs_df['Cluster'] == cluster_num][['Full Name','Speed', 'Acceleration', 'Strength', 'Agility', 'Awareness', 'Carrying', 'Throw Power', 'Break Tackle', 'Injury','Stamina', 'Toughness', 'Break Sack', 'Throw Under Pressure', 'Throw Accuracy Short', 'Throw Accuracy Mid', 'Throw Accuracy Deep', 'Play Action']].head(5))



