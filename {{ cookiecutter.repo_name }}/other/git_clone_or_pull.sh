#!/usr/bin/env bash

if cd $2; then
    git pull;
else
    git clone $1 $2;
fi
cd $2;
python setup.py develop;
