# GymBoard - A small wrapper class that lets you render render OpenAI Gym envs in TensorBoard
<img src="https://raw.githubusercontent.com/mishig25/GymBoard/master/viz.gif" style="width:500px;"/>

### Installation:
```bash
git clone https://github.com/mishig25/GymBoard.git
cd GymBoard
python setup.py sdist
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

Go [here](https://github.com/mishig25/GymBoard/blob/master/tutorial.ipynb) for more comprehensive example.
