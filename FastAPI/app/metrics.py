from prometheus_client import Gauge

# --- PROMETHEUS GAUGES ---

accuracy_gauge = Gauge(
    'model_accuracy',
    'Current accuracy of the prediction model'
)

# Data drift metrics using embeddings
drift_score_gauge = Gauge(
    'drift_score',
    'Combined data drift score based on embedding analysis'
)

semantic_drift_gauge = Gauge(
    'semantic_drift_ks',
    'Kolmogorov-Smirnov statistic for semantic similarity drift'
)

centroid_drift_gauge = Gauge(
    'centroid_drift',
    'Euclidean distance between embedding centroids'
)

spread_drift_gauge = Gauge(
    'spread_drift',
    'Difference in embedding spread/variance between datasets'
)
