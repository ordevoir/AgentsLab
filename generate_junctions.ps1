# generate_junctions.ps1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$yamlPath = Join-Path $repoRoot "context_sources.yaml"
$junctionRoot = Join-Path $repoRoot "junctions"

if (!(Test-Path $yamlPath)) {
  throw "Config not found: $yamlPath"
}

# Простенький парсер YAML для формата:
# sources:
#   - path
#   - path
function Get-SourcesFromYaml([string]$path) {
  $lines = Get-Content -LiteralPath $path
  $inSources = $false
  $sources = New-Object System.Collections.Generic.List[string]

  foreach ($raw in $lines) {
    $line = $raw.Trim()

    if ($line -eq "" -or $line.StartsWith("#")) { continue }

    if ($line -match '^\s*sources\s*:\s*$') {
      $inSources = $true
      continue
    }

    if ($inSources) {
      if ($line -match '^\-\s*(.+)\s*$') {
        $val = $Matches[1].Trim()
        $val = $val.Trim("'").Trim('"')
        if ($val -ne "") { $sources.Add($val) }
        continue
      }

      # Если дошли до другой секции — перестаём читать sources
      if ($line -match '^[A-Za-z0-9_\-]+\s*:') {
        break
      }
    }
  }

  return $sources
}

function Is-ReparsePoint([string]$p) {
  try {
    $item = Get-Item -LiteralPath $p -Force
    return [bool]($item.Attributes -band [IO.FileAttributes]::ReparsePoint)
  } catch {
    return $false
  }
}

function Resolve-TargetPath([string]$repoRoot, [string]$relOrAbs) {
  # Если путь относительный — делаем абсолютным от корня репо
  $candidate = $relOrAbs
  if (-not [IO.Path]::IsPathRooted($candidate)) {
    $candidate = Join-Path $repoRoot $candidate
  }

  # Resolve-Path падает, если пути нет — это хорошо (раньше узнаем об ошибке)
  return (Resolve-Path -LiteralPath $candidate).Path
}

New-Item -ItemType Directory -Path $junctionRoot -Force | Out-Null

$sources = @(Get-SourcesFromYaml $yamlPath)
if ($sources.Count -eq 0) {
  Write-Host "No sources found in context_sources.yaml"
  exit 0
}

# Собираем список ожидаемых имён junction’ов
$expectedNames = New-Object System.Collections.Generic.HashSet[string]

foreach ($src in $sources) {
  $targetAbs = Resolve-TargetPath $repoRoot $src

  $name = Split-Path $src -Leaf
  if ([string]::IsNullOrWhiteSpace($name)) {
    throw "Cannot derive junction name from path: $src"
  }

  if (-not $expectedNames.Add($name)) {
    throw "Name collision: '$name' appears more than once (path: $src). Use unique leaf names."
  }

  $junctionPath = Join-Path $junctionRoot $name

  if (Test-Path -LiteralPath $junctionPath) {
    if (Is-ReparsePoint $junctionPath) {
      Remove-Item -LiteralPath $junctionPath -Force
    } else {
      Write-Warning "Skipping '$junctionPath' because it exists and is NOT a link/reparse-point."
      continue
    }
  }

  New-Item -ItemType Junction -Path $junctionPath -Target $targetAbs | Out-Null
  Write-Host "Created junction: $junctionPath -> $targetAbs"
}

# (Опционально) Чистим "лишние" ссылки в junctions\, которых нет в yaml
Get-ChildItem -LiteralPath $junctionRoot -Force | ForEach-Object {
  $p = $_.FullName
  $n = $_.Name
  if ((Is-ReparsePoint $p) -and (-not $expectedNames.Contains($n))) {
    Remove-Item -LiteralPath $p -Force
    Write-Host "Removed stale junction: $p"
  }
}

Write-Host "All done."
