# Import all functions from the separated service modules for backward compatibility

# State management
from .state import (
    app_state,
    MODEL_NAME,
    MODEL_ALIAS,
    WINDOW_SIZE,
    get_dashboard_data
)

# Model management
from .model_manager import (
    load_and_cache_model,
    get_latest_trained_model,
    promote_model_to_prod
)

# Training pipeline
from .training_pipeline import (
    manage_retraining_data,
    trigger_training_run,
    run_retraining_pipeline
)

# Data management
from .data_manager import (
    write_data_to_csv,
    generate_and_save_gemini_data
)

# Metrics calculation
from .metrics_calculator import (
    calculate_drift_metrics,
    calculate_model_accuracy,
    calculate_and_set_all_metrics
)

# Re-export all functions to maintain backward compatibility
__all__ = [
    # State
    'app_state',
    'MODEL_NAME',
    'MODEL_ALIAS',
    'WINDOW_SIZE',
    'get_dashboard_data',
    
    # Model management
    'load_and_cache_model',
    'get_latest_trained_model',
    'promote_model_to_prod',
    
    # Training pipeline
    'manage_retraining_data',
    'trigger_training_run',
    'run_retraining_pipeline',
    
    # Data management
    'write_data_to_csv',
    'generate_and_save_gemini_data',
    
    # Metrics
    'calculate_drift_metrics',
    'calculate_model_accuracy',
    'calculate_and_set_all_metrics',
]
