@echo off
rem 课程书架的命令行入口：用随附的 Python 运行时运行 lec，用法与开发版相同（如 lec models、lec run 课程名）
setlocal
set PYTHONNOUSERSITE=1
"%~dp0runtime\python.exe" -m lecture_analyzer.cli %*
