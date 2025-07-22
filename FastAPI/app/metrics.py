# 225150200111004_1 HAIKAL THORIQ ATHAYA_1
# 225150200111008_2 MUHAMMAD ARSYA ZAIN YASHIFA_2
# 225150201111001_3 JAVIER AAHMES REANSYAH_3
# 225150201111003_4 MUHAMMAD HERDI ADAM_4

from prometheus_client import Gauge

accuracy_gauge = Gauge(
    'model_accuracy',
    'Current accuracy of the prediction model'
)

# Data drift metrics
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
