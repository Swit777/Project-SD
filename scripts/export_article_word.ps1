param(
    [Parameter(Mandatory = $true)][string]$InputDocx,
    [Parameter(Mandatory = $true)][string]$OutputPdf,
    [ValidateRange(1, 1000)][int]$MaxPages = 10
)

$ErrorActionPreference = 'Stop'
$sourcePath = (Resolve-Path -LiteralPath $InputDocx).Path
$targetPath = [System.IO.Path]::GetFullPath($OutputPdf)
$null = [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($targetPath))
$wordInstance = $null
$articleDocument = $null
try {
    $wordInstance = New-Object -ComObject Word.Application
    $wordInstance.Visible = $false
    $wordInstance.DisplayAlerts = 0
    $wordInstance.AutomationSecurity = 3
    $articleDocument = $wordInstance.Documents.Open($sourcePath, $false, $true, $false)
    $articleDocument.Repaginate()
    $pageCount = $articleDocument.ComputeStatistics(2)
    if ($pageCount -gt $MaxPages) {
        throw "The article has $pageCount pages; the maximum is $MaxPages. Shorten it before exporting."
    }
    $articleDocument.ExportAsFixedFormat($targetPath, 17)
    [PSCustomObject]@{ Pdf = $targetPath; Pages = $pageCount; MaxPages = $MaxPages } | ConvertTo-Json -Compress
}
finally {
    $saveChoice = 0
    try {
        if ($null -ne $articleDocument) {
            $articleDocument.Close([ref]$saveChoice)
            [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($articleDocument)
        }
    }
    finally {
        if ($null -ne $wordInstance) {
            $wordInstance.Quit([ref]$saveChoice)
            [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordInstance)
        }
    }
}
