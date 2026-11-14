
from sklearn.ensemble import ExtraTreesClassifier

from model_utils import run_model

model = ExtraTreesClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

run_model(model, "Extra_Trees_25")
