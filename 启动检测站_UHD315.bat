@echo off
call E:\Anaconda\Scripts\activate.bat n310-uhd315
cd /d C:\Users\98272\Desktop\USRP
python usrp_detector\drone_rf_station_app.py
pause
