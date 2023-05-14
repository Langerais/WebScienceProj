import json

import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import powerlaw
from nltk.corpus import stopwords
from scipy.stats import linregress

# Question 1.1

# Read JSON files
tweets = []
for i in range(1, 4):
    with open(f'tweets{i}.json', 'r', encoding='utf-8') as file:
        for line in file:
            tweets.append(json.loads(line.strip()))

# Create directed user-to-user network
G = nx.DiGraph()
for tweet in tweets:
    user = tweet['user']['id']
    G.add_node(user)

    # Add edges for retweets, mentions, and replies
    if 'retweeted_status' in tweet:
        retweeted_user = tweet['retweeted_status']['user']['id']
        G.add_edge(user, retweeted_user)
    if 'in_reply_to_user_id' in tweet and tweet['in_reply_to_user_id']:
        G.add_edge(user, tweet['in_reply_to_user_id'])
    for mention in tweet['entities']['user_mentions']:
        G.add_edge(user, mention['id'])

# Save network
nx.write_gpickle(G, 'network-lastname.data')

# Network statistics
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges()
density = nx.density(G)
avg_clustering_coefficient = nx.average_clustering(G)
print(f'Number of nodes: {n_nodes}')
print(f'Number of edges: {n_edges}')
print(f'Density: {density}')
print(f'Average clustering coefficient: {avg_clustering_coefficient}')

# Degree distribution
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
degree_counts = np.array([(x, degree_sequence.count(x)) for x in set(degree_sequence)])

# Filter out rows with zeros for plotting and regression
degree_counts = degree_counts[degree_counts[:, 0] != 0]

# Linear regression
slope, intercept, r_value, p_value, std_err = linregress(np.log10(degree_counts[:, 0]), np.log10(degree_counts[:, 1]))

# Plot the degree distribution on a log-log scale
plt.scatter(np.log10(degree_counts[:, 0]), np.log10(degree_counts[:, 1]), label='Data')

# Add the power-law fit line (red slope) to the plot
plt.plot(np.log10(degree_counts[:, 0]), slope * np.log10(degree_counts[:, 0]) + intercept, color='r', label='Fit')

# Format the plot
plt.xlabel('log10(Degree)')
plt.ylabel('log10(Frequency)')
plt.title(f'Degree Distribution (R-squared: {r_value ** 2:.2f})')
plt.legend()
plt.show()

print(f'Slope: {slope}, R-squared: {r_value ** 2}')

# Fit the power-law model to the data
fit = powerlaw.Fit(np.array(degree_counts[:, 1]), discrete=True)

# KS test
ks_stat = fit.power_law.D
p_value = fit.power_law.KS()

print(f"KS statistic: {ks_stat}, p-value: {p_value}")

# Question 1.2
# Calculate centrality measures
# WARNING: This part takes a long time to run
G = nx.read_gpickle('network-lastname.data')

closeness_centrality = nx.closeness_centrality(G)
pagerank = nx.pagerank(G)
betweenness_centrality = nx.betweenness_centrality(G)


def get_screen_name(user_id, tweets):
    for tweet in tweets:
        if tweet['user']['id'] == user_id:
            return tweet['user']['screen_name']
    return None


# Save the results in descending order of centrality in csv

closeness_df = pd.DataFrame(
    [(user_id, get_screen_name(user_id, tweets), closeness) for user_id, closeness in closeness_centrality.items()],
    columns=["user_id", "screen_name", "closeness"])
closeness_df = closeness_df.sort_values("closeness", ascending=False)
closeness_df.to_csv("degree.data", index=False)

pagerank_df = pd.DataFrame([(user_id, get_screen_name(user_id, tweets), rank) for user_id, rank in pagerank.items()],
                           columns=["user_id", "screen_name", "page_rank"])
pagerank_df = pagerank_df.sort_values("page_rank", ascending=False)
pagerank_df.to_csv("pageRank.data", index=False)

betweenness_df = pd.DataFrame([(user_id, get_screen_name(user_id, tweets), betweenness) for user_id, betweenness in
                               betweenness_centrality.items()], columns=["user_id", "screen_name", "betweenness"])
betweenness_df = betweenness_df.sort_values("betweenness", ascending=False)
betweenness_df.to_csv("betweeness.data", index=False)


# Question 1.3


def plot_centrality(file_path, centrality_type):
    data = pd.read_csv(file_path)
    data['screen_name'] = data['screen_name'].astype(str)
    data = data[data['screen_name'] != 'nan']  # Filter out 'nan' users to prevent errors

    top_users = data.head(10).to_dict(orient='records')  # Save top 10 users in a list of dictionaries
    top_10_data = data.head(10).reset_index(drop=True)
    print(f"Top 10 Users by {centrality_type.capitalize()} Centrality:")
    print(top_10_data)

    plt.figure(figsize=(10, 5))
    plt.bar(data['screen_name'][:10], data[centrality_type][:10])
    plt.xticks(rotation=45)
    plt.xlabel('User Screen Name')
    plt.ylabel(f'{centrality_type.capitalize()} Centrality')
    plt.title(f'Top 10 Users by {centrality_type.capitalize()} Centrality')
    plt.show()

    return top_users


top_closeness_users = plot_centrality("degree.data", "closeness")
top_pagerank_users = plot_centrality("pageRank.data", "page_rank")
top_betweenness_users = plot_centrality("betweeness.data", "betweenness")

tweets = []

for i in range(1, 4):
    with open(f'tweets{i}.json', 'r', encoding='utf-8') as file:
        for line in file:
            tweets.append(json.loads(line.strip()))

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


def characterize_users(top_users, tweets, num_tags=5):
    user_tags = {}
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {'http', 'rt', 'worldcup2014'}
    stop_words = stop_words.union(custom_stopwords)

    for user in top_users:
        user_id = user['user_id']
        screen_name = user['screen_name']
        user_tweets = [tweet['text'] for tweet in tweets if tweet['user']['id'] == user_id]

        # Tokenize tweets
        words = nltk.word_tokenize(" ".join(user_tweets).lower())
        words = [word for word in words if word.isalnum() and word not in stop_words]

        tagged_words = nltk.pos_tag(words)
        relevant_tags = ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ']
        relevant_words = [word for word, tag in tagged_words if tag in relevant_tags]

        # Find the most common words
        word_counter = Counter(relevant_words)
        common_words = [word for word, _ in word_counter.most_common(num_tags)]

        user_tags[screen_name] = common_words
        print(f"{screen_name:<15} {' | '.join(common_words)}")

    return user_tags


print('\nTop 10 Users by Closeness Centrality characterized:')
top_closeness_characterized = characterize_users(top_closeness_users, tweets)
print('\nTop 10 Users by Pagerank Centrality characterized:')
top_pagerank_characterized = characterize_users(top_pagerank_users, tweets)
print('\nTop 10 Users by Betweenness Centrality characterized:')
top_betweenness_characterized = characterize_users(top_closeness_users, tweets)

# %%
# Question 2.1

import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
from collections import Counter
from scipy.stats import linregress
import json

# Load the user-to-user network
G = nx.read_gpickle('network-lastname.data')

# Convert the directed graph to an undirected graph for community detection
G_undirected = G.to_undirected()

# Compute the best partition using the Louvain method
partition = community_louvain.best_partition(G_undirected)

communities = {}
for user, community_id in partition.items():
    if community_id not in communities:
        communities[community_id] = []
    communities[community_id].append(user)

# Print the communities with more than 3 users
communities_with_more_than_1_user = {}
for community_id, members in communities.items():
    if len(members) > 1:
        communities_with_more_than_1_user[community_id] = members
        # print(f'Community {community_id}: {len(members)} members')

# Sort communities by the number of users in descending order
sorted_communities = sorted(communities_with_more_than_1_user.items(), key=lambda x: len(x[1]), reverse=True)

# Extract the size of each community
community_sizes = [len(members) for _, members in sorted_communities]

# Fit a power-law distribution to the community sizes
fit = powerlaw.Fit(community_sizes, discrete=True)

# Perform the KS test
ks_stat, ks_pvalue = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)

# Print the results
print(f'Kolmogorov-Smirnov statistic: {ks_stat}, p-value: {ks_pvalue}')

# Calculate the frequency of each community size
size_counts = np.unique(community_sizes, return_counts=True)

# Perform linear regression on the log-log plot of the community size distribution
slope, intercept, r_value, p_value, std_err = linregress(np.log10(size_counts[0]), np.log10(size_counts[1]))

# Print the results
print(f'Slope: {slope}, R-squared: {r_value ** 2}')

# Plot the community size distribution on a log-log scale
plt.scatter(np.log10(size_counts[0]), np.log10(size_counts[1]), label='Data')
plt.plot(np.log10(size_counts[0]), slope * np.log10(size_counts[0]) + intercept, color='r', label='Fit')

# Format the plot
plt.xlabel('log10(Community Size)')
plt.ylabel('log10(Frequency)')
plt.title(f'Community Size Distribution (R-squared: {r_value ** 2:.2f})')
plt.legend()
plt.show()


counter = 0

# Count the total number of communities with more than 1 user
total_communities_more_than_one_user = len(communities_with_more_than_1_user)

# Count the occurrences of each community size
community_size_counts = Counter(community_sizes)

# Define the range boundaries
ranges = [
    (2, 3, '2-3'),
    (4, 10, '4-10'),
    (11, 50, '11-50'),
    (51, 100, '51-100'),
    (101, 500, '101-500'),
    (501, 1000, '501-1000'),
    (1001, 1500, '1001-1500'),
    (1501, 2000, '1501-2000'),
    (2001, float('inf'), '2000+')
]

# Initialize a dictionary to count the occurrences of each range
range_counts = {label: 0 for _, _, label in ranges}

# Count the occurrences of each range
for size in community_sizes:
    for min_size, max_size, label in ranges:
        if min_size <= size <= max_size:
            range_counts[label] += 1
            break

# Print the number of communities grouped by the ranges in a tabular format
print(f"{'Range':<10} | {'Count':<5}")
print("-" * 18)
for label, count in range_counts.items():
    print(f"{label:<10} | {count:<5}")

# Calculate the modularity score
modularity_score = community_louvain.modularity(partition, G_undirected)
print(f'Modularity score: {modularity_score}')

print('Total number of communities with more than 1 user:', total_communities_more_than_one_user)


# Question 2.2

def extract_hashtags(entities):
    return [hashtag['text'] for hashtag in entities['hashtags']]


# Load tweets
tweets = []
for i in range(1, 4):
    with open(f'tweets{i}.json', 'r', encoding='utf-8') as file:
        for line in file:
            tweets.append(json.loads(line.strip()))

print("Tweets loaded")

# Get the top-5 biggest communities
top_5_communities = sorted_communities[:5]

# Dictionary to store hashtags and their frequencies for each community
community_hashtags = {}

# Iterate through the top-5 biggest communities
for community_id, members in top_5_communities:
    hashtags = []

    # Iterate through the members of each community
    for user in members:
        # Extract the tweets of the current user
        user_tweets = [tweet for tweet in tweets if tweet['user']['id'] == user]

        # Extract hashtags from the user's tweets and add them to the list
        for tweet in user_tweets:
            hashtags.extend(extract_hashtags(tweet['entities']))

    # Count the frequency of each hashtag in the community
    hashtag_counts = Counter(hashtags)

    # Store the top hashtags for the current community
    community_hashtags[community_id] = hashtag_counts.most_common(
        5)  # Change the number to get more or fewer top hashtags

    print(f'Community {community_id}: {len(members)} members')

# Print the top-5 biggest communities and their top hashtags
print("Top 5 Communities and Their Top Hashtags")
print(f"{'Community ID':12} | Top Hashtags")
print('-' * 60)
for community_id, top_hashtags in community_hashtags.items():
    hashtags_str = ', '.join([f'#{hashtag} ({count})' for hashtag, count in top_hashtags])
    print(f'{community_id:12} | {hashtags_str}')


# Question 2.3
# Function to calculate the average degree of nodes within a community
def community_avg_degree(community_members, graph):
    edge_count = 0
    for member in community_members:
        for neighbor in graph.neighbors(member):
            if neighbor in community_members:
                edge_count += 1
    return edge_count / len(community_members)


# Calculate the average degree and size for each community
community_avg_degrees_and_sizes = []
for community_id, members in sorted_communities:
    avg_degree = community_avg_degree(members, G)
    community_size = len(members)
    community_avg_degrees_and_sizes.append((community_id, avg_degree, community_size))

# Sort communities by average degree in descending order
sorted_community_avg_degrees_and_sizes = sorted(community_avg_degrees_and_sizes, key=lambda x: x[1], reverse=True)

# Get the top-10 most connected communities
top_10_connected_communities = sorted_community_avg_degrees_and_sizes[:10]

# Print the top-10 most connected communities with the number of users
print("Top 10 Most Connected Communities")
print("---------------------------------")
print("Community | Average Degree | Number of Users")
print("----------------------------------------------")
for community_id, avg_degree, community_size in top_10_connected_communities:
    print(f'{community_id:9} | {avg_degree:14.2f} | {community_size:14}')

# %%
# Question 3

import networkx as nx
import numpy as np
from node2vec import Node2Vec
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import random


def sample_non_edges(graph, num_samples):
    non_edges = []
    nodes = list(graph.nodes())

    while len(non_edges) < num_samples:
        u, v = random.sample(nodes, 2)
        if not graph.has_edge(u, v):
            non_edges.append((u, v))

    return non_edges


# Load your graph
G = nx.read_gpickle('network-lastname.data')

# Generate node embeddings using Node2Vec
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
print("Node2Vec model created")
model = node2vec.fit(window=10, min_count=1, batch_words=4)
print("Node embeddings generated")

# Generate X and y from the node embeddings
X, y = [], []
existing_edges = list(G.edges())
num_existing_edges = len(existing_edges)
non_existing_edges = sample_non_edges(G, num_existing_edges)
print("Non-existing edges sampled")

selected_edges = existing_edges + non_existing_edges

for u, v in selected_edges:
    X.append(model.wv[str(u)] * model.wv[str(v)])
    y.append(1 if G.has_edge(u, v) else 0)
print("X and y generated")

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets")

# Train a custom MLP classifier
clfANN = MLPClassifier(solver='adam', activation='relu',
                       batch_size=10,
                       tol=1e-7,
                       validation_fraction=0.2,
                       hidden_layer_sizes=(15, 10), random_state=1, max_iter=50000, verbose=False)
print("MLP classifier created")
clfANN.fit(X_train, y_train)
print("MLP classifier trained")

# Get predictions on the train and test sets
y_train_pred_ANN = clfANN.predict(X_train)
y_test_pred_ANN = clfANN.predict(X_test)
print("Predictions made")

# Calculate confusion matrices
confMatrixTrainANN = confusion_matrix(y_train, y_train_pred_ANN)
confMatrixTestANN = confusion_matrix(y_test, y_test_pred_ANN)
print("Confusion matrices calculated")

# Print confusion matrices
print('\nConfusion matrix, Train Set, Neural Net')
print(confMatrixTrainANN)
print()

print('Confusion matrix, Test Set, Neural Net')
print(confMatrixTestANN)
print()

# Measures of performance: Precision, Recall, F1
print('0: no link,   1: a link exists')
print(classification_report(y_test, y_test_pred_ANN))
