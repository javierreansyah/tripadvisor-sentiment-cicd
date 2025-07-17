from prometheus_client import Gauge

# --- PROMETHEUS GAUGES ---

accuracy_gauge = Gauge(
    'model_accuracy',
    'Current accuracy of the prediction model'
)

KS_gauge = Gauge(
    'ks_statistic',
    'Kolmogorov-Smirnov statistic for data drift'
)

Wasserstein_gauge = Gauge(
    'wasserstein_distance',
    'Wasserstein distance for data drift'
)
