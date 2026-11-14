
from sklearn.ensemble import AdaBoostClassifier

from model_utils import run_model

model = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)

run_model(model, "AdaBoost_25")
