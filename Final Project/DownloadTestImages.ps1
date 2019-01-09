$csv = Import-Csv -Path "D:\Images\test.csv"

$max = $csv.count - 1

for($i = 0; $i -le $max; $i++){
    $urlToDownload =  $csv[$i].url

    $imageId = $csv[$i].id

    Invoke-WebRequest -Uri $urlToDownload -OutFile "D:\Images\Test\$imageId.jpg"    
}

#for($i = 1; $i -le 100; $i++){

#    $download = $false

#    while (!$download){
#        try{
#            $randNum = Get-Random -Maximum $max


            
#            $download = $true   
#        }
#        catch{}
#    }
#}