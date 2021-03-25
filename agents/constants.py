SpiderAndFlyEnv = 'PredatorPrey10x10-v4'
BaselineModelPath_10x10_4v2 = '../artifacts/baseline_policy_10x10_4v2.pt'
RolloutModelPath_10x10_4v2 = '../artifacts/rollout_policy_10x10_4v2.pt'


class AgentType:
    RANDOM = 'Random Agent'
    RULE_BASED = 'Rule Based'  # Smallest Manhattan Distance
    EXACT_ROLLOUT = 'ExactRollout'

