
from sklearn.tree import DecisionTreeClassifier

from model_utils import run_model

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=20,
    random_state=42
)

run_model(model, "Decision_Tree_25")
