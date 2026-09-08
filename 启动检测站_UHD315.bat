@echo off
call E:\Anaconda\Scripts\activate.bat n310-uhd315
cd /d C:\Users\98272\Desktop\USRP
python s3r_detector_software\drone_rf_station_app.py
pause
