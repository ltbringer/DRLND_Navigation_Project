import numpy as np
import matplotlib.pyplot as plt
from environment.banana_floor import BananaEnv
from agent.banana_collector import Agent
from trainer.train import dqn
from utils.cli.args import fetch_args
from utils.cli.validator import ValidatedArgs


args            = fetch_args()
validated_args  = ValidatedArgs(args)
episodes        = validated_args.episodes()
time_steps      = validated_args.time_steps()
score_window    = validated_args.score_window()
seed            = validated_args.seed()
qualify_score   = validated_args.qualify_score()
buffer_size     = validated_args.buffer_size()
batch_size      = validated_args.batch_size()
gamma           = validated_args.gamma()
eps_start       = validated_args.eps_start()
eps_end         = validated_args.eps_end()
eps_decay       = validated_args.eps_decay()
tau             = validated_args.tau()
lr              = validated_args.lr()
fc1_units       = validated_args.fc1_units()
fc2_units       = validated_args.fc2_units()
update_every    = validated_args.update_every()

env             = BananaEnv(validated_args.env_path())
state_size      = env.get_state_size()
action_size     = env.get_action_size()

agent = Agent(
    state_size,
    action_size,
    seed,
    fc1_units=fc1_units,
    fc2_units=fc2_units,
    buffer_size=buffer_size,
    batch_size=batch_size,
    gamma=gamma,
    tau=tau,
    lr=lr,
    update_every=update_every
)

scores = dqn(
    agent,
    env,
    qualify_score=qualify_score,
    episodes=episodes,
    max_t=time_steps,
    eps_start=eps_start,
    eps_end=eps_end,
    eps_decay=eps_decay
)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
