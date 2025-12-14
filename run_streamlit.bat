@echo off
REM Streamlit Inference Interface Launcher for Windows

echo ==============================================
echo Streamlit HDR Enhancement Interface
echo ==============================================
echo.

echo Starting Streamlit app...
echo.
echo The app will open in your browser at: http://localhost:8501
echo.

streamlit run streamlit_inference.py --server.port 8501 --server.address 0.0.0.0

pause

