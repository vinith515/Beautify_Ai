@echo off
echo Opening Beautify AI Project Report...
if exist "%~dp0Beautify_AI_Project_Report_Final.pdf" (
    start "" "%~dp0Beautify_AI_Project_Report_Final.pdf"
) else (
    start "" "%~dp0Claude_Acne_Model_Project_Report.pdf"
)
