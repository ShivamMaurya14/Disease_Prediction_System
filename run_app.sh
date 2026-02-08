#!/bin/bash

# Explicitly use the python from miniconda to run streamlit
# This avoids needing to 'activate' the environment in the shell script

/opt/miniconda3/bin/python3 -m streamlit run app.py
