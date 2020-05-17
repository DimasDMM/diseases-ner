#!/bin/bash

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source ${SCRIPT_PATH}/environment-vars.sh

if [ $# == 0 ]
then
    idocker python_${PROJECT_USER} bash
elif [ $1 == 'train' ]
then
    idocker python_${PROJECT_USER} python ./code/main.py -t "$2"
elif [ $1 == 'predict' ]
then
    idocker python_${PROJECT_USER} python ./code/main.py -p "$2" "$3"
elif [ $1 == 'init' ]
then
    idocker python_${PROJECT_USER} python ./code/main.py -i
elif [ $1 == 'scrapper' ]
then
    idocker python_${PROJECT_USER} python ./code/main.py -s
fi
