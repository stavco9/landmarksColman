$trainPath = "D:\Images\Train"
$validationPath = "D:\Images\Validation"

ls $trainPath | foreach{
    if ($_.Name -like "*.csv"){
        $csv = Get-Content -Path $_.FullName

        $max = $csv.Count - 1

        $classNum = $_.Name.Split(".")[0]

        New-Item -ItemType Directory -Path "$validationPath\$classNum"

        for($i = 1; $i -le 30; $i++){
                $download = $false

                while (!$download){
                    try{
                        $randNum = Get-Random -Maximum $max

                        $urlToDownload =  $csv[$randNum].split(",")[1]

                        $imageId = $csv[$randNum].split(",")[0]

                        Invoke-WebRequest -Uri $urlToDownload -OutFile "$validationPath\$classNum\$imageId.jpg"
            
                        $download = $true   
                    }
                    catch{}
            }
        }
    }
}