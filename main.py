import numpy as np
import pandas as pd
from t2f.extraction.extractor import feature_extraction
from t2f.utils.importance_old import feature_selection
from t2f.model.clustering import ClusterWrapper

if __name__ == '__main__':
    # 10 multivariate time series with 100 timestamps and 3 signals each
    arr = np.random.randn(10, 100, 3)
    arr[5:] = arr[5:] * 100

    labels = {}  # unsupervised mode
    labels = {0: 'a', 1: 'a', 5: 'b', 6: 'b'}  # semi-supervised mode
    n_clusters = 2  # Number of clusters

    transform_type = 'std'  # preprocessing step
    model_type = 'KMeans'  # clustering model

    # Feature extraction
    df_feats = feature_extraction(arr, batch_size=100, p=1)

    external_feat = df_feats.iloc[:, :2]
    external_feat.columns = ["Lat", "Lon"]

    # Feature selection
    context = {'model_type': model_type, 'transform_type': transform_type,'score_mode':'simple','strategy':'sk_base',
               'k_best':False, 'pre_transform':False, 'top_k':[1], 'pfa_value':0.9}
    top_feats = feature_selection(df_feats, labels=labels, context=context, external_feat=external_feat)

    df_feats = df_feats[top_feats]
    df_feats = pd.concat([df_feats, external_feat], axis=1)

    # Clustering
    model = ClusterWrapper(n_clusters=n_clusters, model_type=model_type, transform_type=transform_type)
    y_pred = model.fit_predict(df_feats)
    print(y_pred)
