SpiderAndFlyEnv = 'PredatorPrey10x10-v4'
BaselineModelPath = '../artifacts/baseline_policy.pt'


class ModelType:
    RANDOM = 'Random Agent'
    BASELINE = 'Baseline'  # Smallest Manhattan Distance
