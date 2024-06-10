SpiderAndFlyEnv = 'PredatorPrey10x10-v4'
BaselineModelPath_10x10_4v2 = 'artifacts/baseline_policy_10x10_4v2.pt'
RolloutModelPath_10x10_4v2 = 'artifacts/rollout_policy_10x10_4v2.pt'
RepeatedRolloutModelPath_10x10_4v2 = 'artifacts/repeated_rollout_policy_10x10_4v2.pt'


class AgentType:
    RANDOM = 'Random'
    RULE_BASED = 'Rule-Based'  # Smallest Manhattan Distance
    QNET_BASED = 'QNet-Based'
    SEQ_MA_ROLLOUT = 'Agent-by-agent MA Rollout'
    STD_MA_ROLLOUT = 'Standard MA Rollout'


class QnetType:
    BASELINE = 'Trained from Rollout Data'
    REPEATED = 'Trained from Repeated Rollout Data'
