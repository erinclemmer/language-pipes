env/bin/coverage run -m pytest tests --ignore=tests/distributed_state_network
env/bin/coverage report -m --omit="/tmp/*,src/language_pipes/commands/*,tests/*,src/language_pipes/util/user_prompts.py,src/language_pipes/distributed_state_network/*" --sort=cover
