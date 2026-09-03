@echo off
setlocal

echo [my_tutor] 开始执行视频处理全流程...
python run_pipeline.py %*
set EXIT_CODE=%ERRORLEVEL%

if not %EXIT_CODE%==0 (
  echo.
  echo [my_tutor] 流程执行失败，退出码: %EXIT_CODE%
  pause
  exit /b %EXIT_CODE%
)

echo.
echo [my_tutor] 流程执行成功。
pause
