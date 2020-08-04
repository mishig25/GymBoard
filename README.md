# GymBoard - A small wrapper class that lets you render OpenAI Gym envs (as GIFs) in TensorBoard's Images tab
<img src="https://raw.githubusercontent.com/mishig25/GymBoard/master/viz.gif" style="width:500px;"/>

### Installation:
```bash
pip install git+https://github.com/mishig25/GymBoard
```

### Example usage:
in your python file, call:
```python
import gym
from gymboard import GymBoard

env = gym.make('CartPole-v0')

gboard = GymBoard()
gboard.show()
gboard.write_env(env, step=0)
```

Checkout the notebook [here](https://github.com/mishig25/GymBoard/blob/master/tutorial.ipynb) for more comprehensive example.

License: MIT