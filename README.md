- Posteriors [package](https://github.com/normal-computing/posteriors) and [documentation](https://normal-computing.github.io/posteriors/)

Data source [https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

## Running the Bayesian training
Start a tmux session (recommended for long runs):
```bash
tmux new -s bayesian_training
# inside tmux:
.venv/bin/python scripts/bayesian_training_script.py --sampler SAMPLER_NAME
SAMPLER_NAME can be one of: `vi`, `ekf`, `laplace`, `sgmcmc`
```
## Stopping / managing tmux sessions
- List sessions:
```bash
tmux ls
```
- Kill a session:
```bash
tmux kill-session -t bayesian_training
```
- Attach to a session and stop:
```bash
tmux attach -t bayesian_training
# then inside tmux: Ctrl-D or `exit`
```

The model checkpoints and logs will be saved in the `checkpoints/samplers/SAMPLER_NAME_sampler` folder.