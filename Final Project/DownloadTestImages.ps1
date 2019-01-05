#$csv = Import-Csv -Path C:\DeepLearning\FinalProject\google-landmarks-dataset\test.csv

$max = $csv.count - 1

for($i = 1; $i -le 100; $i++){

    $download = $false

    while (!$download){
        try{
            $randNum = Get-Random -Maximum $max

            $urlToDownload =  $csv[$randNum].url

            $imageId = $csv[$randNum].id

            Invoke-WebRequest -Uri $urlToDownload -OutFile "C:\DeepLearning\FinalProject\Test\$imageId.jpg"
            
            $download = $true   
        }
        catch{}
    }
}