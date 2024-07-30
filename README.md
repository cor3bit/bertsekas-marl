## Multiagent Reinforcement Learning: Rollout and Policy Iteration


Implementation of the Multiagent Rollout
based on the 
paper by Dimitri Bertsekas (2021).


### Environment

Simulation environment follows the rules of the Spiders-and-Flies game as 
specified in [1]. The environment is adapted from Anurag Koul's ma-gym [2] 
modifying the PredatorPrey env.


### Usage

- Install the requirements with `pip`:

```
$ pip install -r requirements.txt
```

- Run the agent simulation from the `scripts` folder:
May have to add the pwd to the path: eg- 
export PYTHONPATH=$PYTHONPATH:/path/to/bertsekas-marl
python3 scripts/run_agent.py
```
$ python run_agent.py

```

- Run agents' comparison from the `scripts` folder:

```
$ python run_comparison.py
```

### Results
Result are published below:
[WANDB.ai](https://wandb.ai/athmajan-university-of-oulu/SecurityAndSurveillance/reports/Multiagent-Reinforcement-Learning-Rollout-and-Policy-Iteration--Vmlldzo4ODYxNDMx?accessToken=myfjbwjdmpdno7dz0ya9s4ty4f58ik9im0sqv3ki0i640qkhet8e818gffb6rw9m)

### References

1. Dimitri Bertsekas - Multiagent Reinforcement Learning: Rollout and 
   Policy Iteration (2021).
   Web: https://ieeexplore.ieee.org/document/9317713
      
2. Anurag Koul - ma-gym: Collection of multi-agent environments based 
   on OpenAI gym (2019). Web: https://github.com/koulanurag/ma-gym
   