import numpy as np
import pandas as pd
from Time2Feat.extraction.extractor import feature_extraction
from Time2Feat.utils.importance_old import feature_selection
from Time2Feat.model.clustering import ClusterWrapper
from Time2Feat.Time2Feat import Time2Feat
if __name__ == '__main__':
    # 10 multivariate time series with 100 timestamps and 3 signals each
    arr = np.random.randn(10, 100, 3)
    arr[5:] = arr[5:] * 100
    external_feat = pd.DataFrame({'LEN': [0 for x in range(arr.shape[0])]})
    # labels = {}  # unsupervised mode
    labels = {0: 'a', 1: 'a', 5: 'b', 6: 'b'}  # semi-supervised mode
    n_clusters = 2  # Number of clusters
    model = Time2Feat(n_clusters)
    print(model.fit_predict(arr, labels=labels, external_feat=external_feat))