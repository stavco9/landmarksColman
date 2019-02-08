$rootPath = "C:\DeepLearning\RealImages - Copy\Validation"

gci $rootPath | foreach{
    $labelPath = ($rootPath + "\" + $_.Name)

    if ((Get-Item $labelPath) -is [System.IO.DirectoryInfo]){
        $i = 1

        gci ($labelPath) | foreach {
            if (($i -le 7)){
                Remove-Item -Path ($labelPath + "\" + $_.Name) -Force
            }
            
            $i++
        }
    }
}