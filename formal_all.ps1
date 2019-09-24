pushd .
Get-ChildItem -Path . -Directory -Exclude libs, utility | 
    foreach {
        cd $_.FullName
        Get-ChildItem -Path . -Directory -Recurse |
        foreach {
            echo $_.FullName
            cd $_.FullName
            If (Test-Path *.cpp){
                &clang-format -i -style=file *.cpp
            }
            If (Test-Path *.h){
                &clang-format -i -style=file *.h
            }
            If (Test-Path *.cu){
                &clang-format -i -style=file *.cu
            }
        }
    }
popd


