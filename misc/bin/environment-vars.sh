#!/usr/bin/env bash

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Including .env file
set -o allexport
source $SCRIPT_PATH/../../.env
set +o allexport

# Fix $USER variable if needed (Git Bash Windows)
if [[ "$OSTYPE" == "msys" ]]; then
	export USER="$(whoami)"
fi

# Add project variable
export PROJECT_USER="${PROJECT_NAME}_${USER}"

# Function to execute Docker commands
function idocker() {
    # $1: container name
    # $2: command, e.g. "sh" or "python"
    # $3: parameter of the previous command

	params=()
	i=0
	for c in "$@"; do
		if [[ $i -lt 4 ]]; then
			params+=("${c}")
		else
			params+=("\"${c}\"")
		fi

		let i=i+1
	done

	if [[ "$OSTYPE" == "msys" ]]; then
	    # Add winpty if Windows Git Bash
		command="winpty docker exec -it ${params[@]}"
	else
		command="docker exec -it ${params[@]}"
	fi

	eval $command
}
