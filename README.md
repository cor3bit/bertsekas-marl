## Multiagent Reinforcement Learning: Rollout and Policy Iteration


Re-implementation of Implementation of the Multiagent Rollout based on the 
paper by Dimitri Bertsekas (2021). Originally forked from https://github.com/cor3bit/bertsekas-marl.
But a few improvements and explorations have been made from there.


### Environment
Simulation environment follows the rules of the Spiders-and-Flies game as 
specified in [1]. The environment is adapted from Anurag Koul's ma-gym [2] 
modifying the PredatorPrey env. A wrapper has been written for a modified environment with 
modified reward functions to test the online replanning discussed at the end of section V in [1].


### Usage

- Install the requirements with `pip`:

```
$ pip install -r requirements.txt
```

`scripts` folder is mostly for the implementation from cor3bit.
All my scripts are present in the main directory.

- Manhattan distance based agent. (Rule based agent)
```
$ python runRuleBasedAgent.py
```

- Run standard rollout. (All agents at once based on simulated Manhattan distance rules)
```
$ python runStandRollout.py
```

- Run sequential rollout. (One agent at a time.)
```
$ python runSeqRollout.py
```

- Learn a rolllout policy network from experiences collected at sequential rollout.
```
$ python learnRolloutOffV2.py
```

- Run Autonomous Rollout using signaling policy and base policy
```
$ python runAutoOffline.py
```


## Results



### References

1. Dimitri Bertsekas - Multiagent Reinforcement Learning: Rollout and 
   Policy Iteration (2021).
   Web: https://ieeexplore.ieee.org/document/9317713
      
2. Anurag Koul - ma-gym: Collection of multi-agent environments based 
   on OpenAI gym (2019). Web: https://github.com/koulanurag/ma-gym
   