
#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

PROJECT_NAME = {{ cookiecutter.repo_name }}
PYTHON_INTERPRETER = {{ cookiecutter.python_interpreter }}

PROFILE = {{ cookiecutter.aws_profile }}
AWS_KEY = {{ cookiecutter.aws_key }} 
AWS_HOST = {{ cookiecutter.aws_host }}
S3_BUCKET = {{ cookiecutter.s3_bucket }}


#################################################################################
# ENVIRONMENT                                                                   #
#################################################################################

pyenv:
	curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
	cat other/to_bashrc_by_pyenv.txt >>  ~/.bashrc
	# Use `source ~/.bashrc` command

python:
	sudo apt-get update
	sudo apt-get upgrade
	sudo apt-get dist-upgrade
	sudo apt-get install build-essential python-dev python-setuptools python-pip python-smbus
	sudo apt-get install libncursesw5-dev libgdbm-dev libc6-dev
	sudo apt-get install zlib1g-dev libsqlite3-dev tk-dev
	sudo apt-get install libssl-dev openssl
	sudo apt-get install libffi-dev
	env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.7.0

venv:
	pyenv virtualenv 3.7.0 $(PROJECT_NAME)
	#
	# Virtualenv $(PROJECT_NAME) ready, run following to activate
	# pyenv activate $(PROJECT_NAME)


requirements:
	# for fasttext it is necessary to install Cython, h5py first
	# some bug, it can't be installed automaticaly
	pip install --upgrade  pip setuptools wheel; \
	python setup.py develop

	./other/git_clone_or_pull.sh https://github.com/seznam/flexp.git ~/src/flexp/
	./other/git_clone_or_pull.sh https://github.com/tomasprinda/tputils.git ~/src/tputils/


#################################################################################
# DATA                                                                          #
#################################################################################

	


#################################################################################
# AWS                                                                           #
#################################################################################

data_to_s3:
	aws s3 sync data/ s3://$(S3_BUCKET)/aukro_categories/data/ --profile $(PROFILE)

data_from_s3:
	aws s3 sync s3://$(S3_BUCKET)/$(PROJECT_NAME)/data/ data/ --profile $(PROFILE)

ssh:
	ssh -i $(AWS_KEY) -t ubuntu@$(AWS_HOST) "cd /home/ubuntu/src/$(PROJECT_NAME); bash"

sshfs:
	sshfs \
    -o follow_symlinks \
    -o transform_symlinks \
    -o TCPKeepAlive=yes \
    -o ServerAliveCountMax=2 \
    -o ServerAliveInterval=5 \
    -o IdentityFile=$(AWS_KEY) \
    ubuntu@$(AWS_HOST):/home/ubuntu/  \
    ~/einstein/

sshfs_disconnect:
	fusermount -u ~/einstein


#################################################################################
# Run services                                                                  #
#################################################################################

run_jupyter:
	jupyter notebook --port 8882 > /dev/null 2> /dev/null

run_flexp:
	cd ~/src/$(PROJECT_NAME)/experiments ; \
    python browse.py --port 7772 > /dev/null 2> /dev/null

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
