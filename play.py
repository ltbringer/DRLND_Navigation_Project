import sys
from environment.banana_floor import BananaEnv
from utils.cli.args import fetch_args
from utils.cli.validator import ValidatedArgs
from agent.banana_collector import Agent


args            = fetch_args()
validated_args  = ValidatedArgs(args, has_model=True)
model_path      = validated_args.model_path()
episodes        = validated_args.episodes()
time_steps      = validated_args.time_steps()
seed            = validated_args.seed()
buffer_size     = validated_args.buffer_size()
batch_size      = validated_args.batch_size()
gamma           = validated_args.gamma()
tau             = validated_args.tau()
lr              = validated_args.lr()
fc1_units       = validated_args.fc1_units()
fc2_units       = validated_args.fc2_units()


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
    lr=lr
)

agent.load_saved_model(model_path)

for i_episode in range(episodes):
    env_info = env.reset_to_initial_state(train_mode=False)
    score = 0
    state = env.get_state_snapshot()
    yellow_bananas = 0
    blue_bananas = 0
    for t in range(time_steps):
        action = agent.act(state, 0)
        next_state, reward, done = env.step(action).reaction()
        if reward == 1:
            yellow_bananas += 1
        elif reward == -1:
            blue_bananas += 1
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        sys.stdout.write("Yellow bananas: %d Blue bananas: %d \r" % (yellow_bananas, blue_bananas))
        sys.stdout.flush()
        if done:
                break
