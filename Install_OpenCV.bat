bitsadmin /TRANSFER DownloadOpenCV https://github.com/opencv/opencv/releases/download/4.11.0/opencv-4.11.0-windows.exe "%~dp0opencv-4.11.0-windows.exe"
"C:/Program Files/7-Zip/7z.exe" x "%~dp0opencv-4.11.0-windows.exe" -o"%~dp0"
pause
