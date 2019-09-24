@echo Cleaning directory ...

del /S /A H *.suo
del /S /A H *.db 
del /S *.log
del /S conanbuildinfo*
del /S conaninfo*

FOR /F "tokens=*" %%G IN ('DIR /B /AD /S Release') DO RD /S /Q "%%G"
FOR /F "tokens=*" %%G IN ('DIR /B /AD /S Debug') DO RD /S /Q "%%G"
FOR /F "tokens=*" %%G IN ('DIR /B /AD /S x64') DO RD /S /Q "%%G"
FOR /F "tokens=*" %%G IN ('DIR /B /AD /S prj') DO RD /S /Q "%%G"

@echo Done.
