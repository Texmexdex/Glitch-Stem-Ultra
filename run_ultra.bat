@echo off
title GLITCH STEM ULTRA
color 0b

echo.
echo                                     s                         
echo               x d88"     @88^>      :8                 uef^^"   
echo                5888R      %%8P      .88                :d88E      
echo        uL       '888R              :888ooo             `888E      
echo    ue888Nc        888R     @88u   -*8888888   udR88N     888E  z8k 
echo    d88E`"888E`     888R   ''888E`    8888   ^<888'888k  888E~?888L
echo    888E   888E     888R     888E    8888   9888 'Y"    888E  888E
echo   888E   888E    888R     888E    8888    9888         888E  888E
echo  888E    888E    888R    888E    8888Lu=  9888         888E  888E
echo  888^&    888E    888B    888^&    %%888*   ?8888u  /    888E  888E
echo  *888"//888^&  ^^*888%%   R888"    'Y"     "8888P'    m888N  888^>
echo   `"    "888E      "%%       ""                "P'      `Y"  888 
echo          `88E                                               J88" 
echo    4888~  J8%%                                              @%%   
echo     ^^"===*"`                                              :"     
echo.
echo        STEM ULTRA // TeXmExDeX Type Tunes
echo.

:: Check if venv exists in current directory
if not exist "venv_ultra\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at venv_ultra
    echo [INFO] Please run setup_ultra.bat first.
    echo.
    pause
    exit /b 1
)

:: Activate and run
call venv_ultra\Scripts\activate
python GlitchStemUltra.py

:: Keep window open if python crashes
if errorlevel 1 (
    echo.
    echo [CRITICAL] Application exited with an error.
    pause
)
