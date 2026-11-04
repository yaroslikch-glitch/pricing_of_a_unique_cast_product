@echo off
chcp 65001 > nul
echo =======================================================
echo Запуск веб-сервиса: Ценообразование уникального литья
echo ЮУрГУ & ООО "БВК" (ПГ КОНАР, г. Челябинск)
echo Ускорение: AMD Radeon RX 6600 (Vulkan NCNN)
echo =======================================================
echo.
cd /d "%~dp0"
call .\venv\Scripts\activate
python web_app.py
pause
