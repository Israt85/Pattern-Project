
from sklearn.ensemble import HistGradientBoostingClassifier

from model_utils import run_model

model = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=42
)

run_model(model, "Gradient_Boosting_25")
