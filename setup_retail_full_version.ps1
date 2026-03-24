# PowerShell helper to scaffold the same folder structure on Windows
param(
    [string]$TargetPath = ".\retail_decision_support_full_version"
)

$folders = @(
    "app", "config", "data\raw", "data\generated", "data\processed", "data\reference",
    "artifacts\runs", "src", "docs", "tests", "outputs", "notebooks"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Force -Path (Join-Path $TargetPath $folder) | Out-Null
}

Write-Host "Folder structure created at $TargetPath"
Write-Host "Copy the starter Python files into src/, config/, and app/ from the packaged project."
