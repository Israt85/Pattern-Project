
from sklearn.ensemble import RandomForestClassifier

from model_utils import run_model

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

run_model(model, "Random_Forest_25")
