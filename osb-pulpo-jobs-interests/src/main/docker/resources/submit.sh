#!/bin/bash

jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 jupyter/interests.ipynb