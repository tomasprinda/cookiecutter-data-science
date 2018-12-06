import os
from os.path import dirname as up

PROJECT_DIR = up(up(os.path.realpath(__file__)))
EXP_DIR = os.path.join(PROJECT_DIR, "experiments")
