param([string]$dir)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null=[Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]
$null=[Windows.Graphics.Imaging.BitmapDecoder,Windows.Graphics.Imaging,ContentType=WindowsRuntime]
$null=[Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime]
$asTaskGeneric=([System.WindowsRuntimeSystemExtensions].GetMethods()|?{$_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'})[0]
function Await($op,$t){$m=$asTaskGeneric.MakeGenericMethod($t);$task=$m.Invoke($null,@($op));$task.Wait();$task.Result}
$eng=[Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Globalization.Language]::new('en-US'))
if(-not $eng){$eng=[Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()}
Get-ChildItem "$dir\*" -Include *.png,*.jpg | Sort-Object Name | % {
  $f=Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($_.FullName)) ([Windows.Storage.StorageFile])
  $s=Await ($f.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
  $dec=Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($s)) ([Windows.Graphics.Imaging.BitmapDecoder])
  $bmp=Await ($dec.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
  $r=Await ($eng.RecognizeAsync($bmp)) ([Windows.Media.Ocr.OcrResult])
  $txt=($r.Lines|%{$_.Text}) -join "`n"
  $txt | Out-File -Encoding utf8 ($_.FullName -replace '\.(png|jpg)$','.txt')
  "{0} words={1}" -f $_.Name, ($txt -split '\s+').Count
}
