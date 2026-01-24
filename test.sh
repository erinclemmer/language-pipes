env/bin/coverage run -m pytest tests
env/bin/coverage report -m --omit="/tmp/*,src/language_pipes/commands/*,tests/*,src/language_pipes/util/user_prompts.py" --sort=cover
