from tensorflow.keras import datasets, Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from validclust import dunn
import cv2


class KMeansClustering:

    def __init__(self, epochs, X, K=10):
        # K is the no. of Clusters to be formed
        self.K = K
        self.epochs = epochs

        # Initializing 10 Empty Clusters as K=10
        self.clusters = [[], [], [], [], [], [], [], [], [], [], []]
        # Initializing Centroids of each Cluster which are the mean feature vector for each cluster
        self.centroids = []

        # Initializing the Data and its sample and feature strength
        self.X = X
        self.no_of_samples, self.no_of_features = X.shape

    # Static Method to Find Euclidean Distance Between Two Given Feature Vectors
    @staticmethod
    def calculateEuclideanDistance(Xa, Xb):
        return np.sqrt(np.sum((Xa - Xb) ** 2))

    # Predicting the Labels of the Samples running the samples
    def predictLabels(self):

        # initialize the Centroid with 10 Random Samples to start forming the Clusters
        random_indexes_in_range = np.random.choice(self.no_of_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_indexes_in_range]

        # Running the Loop for Fixed No. of Epochs or if there is no change in Centroid to Find appropriate Clusters
        for _ in range(self.epochs):
            # Assigning Clusters to Newly Calculated Centroids
            self.clusters = self.clusterFormation(self.centroids)

            # Backing Up Old Centroid to Check for Convergence with Newly Calculate Centroids
            oldCentroids = self.centroids

            # Calculating New Centroids from the Clusters
            self.centroids = self.calculateCentroids(self.clusters)

            # Convergence Check, if Found then no Point in iterating Further
            if self.isConvergenceFound(oldCentroids, self.centroids):
                break

        # Final Clusters are formed and Samples are given the Cluster Labels to which they are assigned
        return self.getClusterLabels(self.clusters)

    # Assign the samples to the closest centroids to create clusters
    def clusterFormation(self, centroids):
        # Initializing 10 Empty Clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            # Finding Euclidean distance of Each Sample with every Centroids
            distances = [self.calculateEuclideanDistance(sample, point) for point in centroids]
            # Fetching Index of Centroid which had Minimum  Distance from Sample
            centroid_idx = np.argmin(distances)
            # Adding to the Cluster with Minimum Centroid Euclidean Distance
            clusters[centroid_idx].append(idx)
        return clusters

    # Assign mean value of clusters to centroids
    def calculateCentroids(self, clusters):
        # Initializing the Centroids List with Zero
        centroids = np.zeros((self.K, self.no_of_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    # Distances between each old and new centroids, for all centroids, if no difference then return True or else False
    def isConvergenceFound(self, oldCentroids, centroids):
        distances = [self.calculateEuclideanDistance(oldCentroids[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    # Each sample will get the label of the cluster it was assigned to
    def getClusterLabels(self, clusters):
        labels = np.empty(self.no_of_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels


class Auto_encoder:

    # Encoding Layers of Auto-Encoder
    def encode(self, input_dim):
        X1 = Dense(128, activation='LeakyReLU')(input_dim) # Hidden Layer 1 of Encoder with 128 Features
        encoded_imgs = Dense(64, activation='LeakyReLU')(X1) # Bottleneck Layer with 64 Features
        return encoded_imgs

    # Decoding Layers of Auto-Encoder
    def decode(self, encoded):
        X2 = Dense(128, activation='LeakyReLU')(encoded) # Hidden Layer 1 of Decoder with 128 Features
        decoded_imgs = Dense(1024, activation='LeakyReLU')(X2) # Final Reconstructed Layer with Full 1024 Features
        return decoded_imgs

    # To compile and Fit the Auto Encoder
    def model_compile_and_fit(self, input_features, encoded_features, decoded_features, X_train):
        auto_encoder = Model(input_features, decoded_features)
        auto_encoder.compile(optimizer=Adam(), loss=BinaryCrossentropy())
        # As the Input and Reconstructed output must be as identical as possible we take both parameters as X_train
        auto_encoder.fit(X_train, X_train, epochs=20, batch_size=256, shuffle=True, verbose=1)
        return X_train

    # To return the Input, Encoded and Decoded Features
    def encode_decode_model(self):
        input_features = Input(shape=(1024,))
        encoded_features = self.encode(input_features)
        decoded_features = self.decode(encoded_features)
        return input_features, encoded_features, decoded_features


# Fetching Train and Test Images for use in Clustering from Cifar 10 Dataset
(X_train, _), (X_test, _) = datasets.cifar10.load_data()

# Converting the Train and Test Images to Grey Scale with Help of CV2
X_train = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
X_test = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

# Normalizing Train and Test Data so that every value is in range [0,1]
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Reshaping Train and Test Dataset Features for each Sample from 32*32 into 1024
X_train = X_train.reshape(50000, 1024)
X_test = X_test.reshape(10000, 1024)

print('===================== Part 1 - K-Means Clustering from Scratch ===================== \n')
# Object Creation of KMeansClustering Class to fit and predict the Cluster Labels of Test Data
k_means = KMeansClustering(25, X_test)

# Fetching Cluster Labels after predicting
cluster_labels = k_means.predictLabels().astype(int)

# Calculating the Quality of Cluster with Cluster Evaluation Metric Silhouette Score
ASC = silhouette_score(X_test, cluster_labels)
print("Average Silhouette Coefficient for Part 1 is ", round(ASC, 4))

# Calculating the Quality of Cluster with Cluster Evaluation Metric Dunn's Index
dist = pairwise_distances(X_test)
dunn_index = dunn(dist, cluster_labels)
print("Dunn's Index for Part 1 is ", round(dunn_index, 4))

print('\n===================== Part 2 - K-Means of Sparse Represented Data with Auto-Encoders ===================== \n')

ae = Auto_encoder()

# Fetching the Input, Encoder and Decoder Features Layers
inp_features, enc_features, dec_features = ae.encode_decode_model()
# Fetch the modified X_train with Auto-Encoder
X_train = ae.model_compile_and_fit(inp_features, enc_features, dec_features, X_train)

# Encoder Model for getting Reduced Dimension Data
encoder = Model(inp_features, enc_features)
# Predicting with Encoded(Sparse Represented) Data
pred = encoder.predict(X_train)

# Inbuilt K Means for 10 Clusters
k_means = KMeans(n_clusters=10)
# Cluster Labels Fetched from K Means Algo from Encoder Predicted Data
cluster_labels = k_means.fit_predict(pred)

# Calculating the Quality of Cluster with Cluster Evaluation Metric Silhouette Score
ASC = silhouette_score(X_train, cluster_labels)
print("Average Silhouette Coefficient for Part 2 is ", round(ASC, 4))
