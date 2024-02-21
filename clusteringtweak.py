import pandas as pd
from sklearn.cluster import OPTICS, DBSCAN
from joblib import Parallel, delayed
import plotly.express as px
from time import perf_counter

min_cluster_size = 100

def cluster_data(file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Extract the "ra" and "dec" columns
    data = df[["ra", "dec"]]

    # Use the OPTICS clustering algorithm
    t1 = perf_counter()
    optics = DBSCAN(eps=1/3600, min_samples=9, n_jobs=-1, algorithm="kd_tree")  # n_jobs=-1 for parallel processing
    df['cluster'] = optics.fit_predict(data)

    df = df.groupby('cluster').filter(lambda x: len(x) > min_cluster_size or (x["w1mpro"].median() < 14 and len(x) > 20))


    print(f"Elapsed -- {perf_counter()-t1}")
    
    # print number of clusters
    print(f"Number of clusters: {len(df['cluster'].unique())}")

    # Plot the clustered data
    fig = px.scatter(df, x="ra", y="dec", color="cluster", title="Clustering on 'ra' and 'dec'")
    #reverse x
    fig.update_xaxes(autorange="reversed")
    fig.write_image("cluster_plot.png")

# Call the function with the path to your CSV file
cluster_data('fdsafs.csv')