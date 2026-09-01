@echo off
REM DSGF-HRL training launcher — UTF-8 safe on Windows GBK terminals
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
E:\ANACONDA\envs\dpg_hrl\python.exe train.py %*
