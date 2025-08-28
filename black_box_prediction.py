import pandas as pd
from scarf_weighting import scarf_weighting
from target_domain import get_target_data

# Get black-box predictions for combined (selected source and target) samples
def predict_target_proba(input_data_target):
    X_target, _, _, _, _, _, rf_model_target = get_target_data()
    input_df_target = pd.DataFrame(input_data_target, columns=X_target.columns)
    return rf_model_target.predict_proba(input_df_target)

def get_black_box_prediction():
    top_per_samples , _ = scarf_weighting()
    y_perturbed_target_pred = predict_target_proba(top_per_samples)
    y_blackbox_target_pred_class = 2 * (y_perturbed_target_pred[:, 1] > 0.5).astype(int) - 1  # Convert probabilities to {-1, 1}
    return y_blackbox_target_pred_class
