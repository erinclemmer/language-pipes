# python -m unittest discover tests/distributed_state_network/
env/bin/coverage run -m pytest tests/distributed_state_network
env/bin/coverage report -m --omit="/tmp/*,src/language_pipes/commands/*,tests/*,src/language_pipes/util/user_prompts.py" --sort=cover