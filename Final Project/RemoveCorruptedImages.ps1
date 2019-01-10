function IsJpegImage
{
param(
[string]
$FileName
)

    try{
        $img = [System.Drawing.Image]::FromFile($filename);
        return $img.RawFormat.Equals([System.Drawing.Imaging.ImageFormat]::Jpeg);
    }
    catch{
        return $false;
    }
}

# Root path of images
$imagesPath = "C:\DeepLearning\FinalProject\Train\"

ls $imagesPath | foreach{

    # If the current item is directory
    if ((Get-Item $_.FullName) -is [System.IO.DirectoryInfo]){

        # The directory represent the class number (landark id)
        $class = $_.FullName

        ls $class | foreach{
            $jpg = $_.FullName

            # If the image can't be resolved with jpg format
            if (!(IsJpegImage -FileName $jpg)){

                # Remove this image
                Remove-Item -Path $_.FullName -Force

                write "Removed $jpg"
            }
        }
    }
}