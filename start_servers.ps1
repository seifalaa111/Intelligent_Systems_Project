# MIDAN server launcher
$wd = $PSScriptRoot
$py = "C:\Python314\python.exe"

function Get-ListenersForPort([int]$Port) {
    netstat -ano | Select-String ":$Port " | ForEach-Object {
        $parts = $_.ToString().Trim() -split '\s+'
        if ($parts.Length -ge 5 -and ($parts -contains "LISTENING")) {
            [PSCustomObject]@{
                Proto   = $parts[0]
                Local   = $parts[1]
                State   = if ($parts.Length -ge 6) { $parts[3] } else { "" }
                Pid     = [int]$parts[-1]
            }
        }
    }
}

function Stop-ProjectProcessOnPort([int]$Port) {
    $expected = switch ($Port) {
        8000 { "-m uvicorn api:api" }
        3000 { "-m http.server 3000" }
        8501 { "-m streamlit run app.py" }
        default { "" }
    }

    $listeners = Get-ListenersForPort $Port | Group-Object Pid | ForEach-Object { $_.Group[0] }
    foreach ($listener in $listeners) {
        if ($listener.Pid -le 0) { continue }
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.Pid)" -ErrorAction SilentlyContinue
        if (-not $proc) { continue }
        $cmd = $proc.CommandLine
        if ($cmd -and $expected -and $cmd.Contains($expected)) {
            Stop-Process -Id $listener.Pid -Force -ErrorAction SilentlyContinue
            continue
        }
        throw "Port $Port is already in use by PID $($listener.Pid): $cmd"
    }
}

function Wait-ForHttp([string]$Url, [int]$TimeoutSec = 30, [switch]$AsJson) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    do {
        try {
            if ($AsJson) {
                return Invoke-RestMethod $Url -TimeoutSec 5
            }
            return Invoke-WebRequest $Url -UseBasicParsing -TimeoutSec 5
        } catch {
            Start-Sleep -Milliseconds 800
        }
    } while ((Get-Date) -lt $deadline)
    throw "Timed out waiting for $Url"
}

Write-Host "Starting MIDAN servers..." -ForegroundColor Cyan

foreach ($port in @(8000, 3000, 8501)) {
    Stop-ProjectProcessOnPort $port
}

Start-Sleep -Seconds 1

$apiArgs = "-m uvicorn api:api --host 0.0.0.0 --port 8000"
$api = Start-Process -FilePath $py -ArgumentList $apiArgs -WorkingDirectory $wd -WindowStyle Hidden -PassThru
Write-Host "  API server PID: $($api.Id) -> http://localhost:8000" -ForegroundColor Green

$httpArgs = "-m http.server 3000"
$http = Start-Process -FilePath $py -ArgumentList $httpArgs -WorkingDirectory $wd -WindowStyle Hidden -PassThru
Write-Host "  Frontend server PID: $($http.Id) -> http://localhost:3000/midan.html" -ForegroundColor Green

$streamlitArgs = "-m streamlit run app.py --server.port 8501"
$streamlit = Start-Process -FilePath $py -ArgumentList $streamlitArgs -WorkingDirectory $wd -WindowStyle Hidden -PassThru
Write-Host "  Streamlit PID: $($streamlit.Id) -> http://localhost:8501" -ForegroundColor Green

try {
    $h = Wait-ForHttp "http://localhost:8000/health" -TimeoutSec 25 -AsJson
    Write-Host "  [OK] API health: models_loaded=$($h.models_loaded)" -ForegroundColor Green
} catch {
    Write-Host "  [FAIL] API not responding" -ForegroundColor Red
}

try {
    $r = Wait-ForHttp "http://localhost:3000/midan.html" -TimeoutSec 20
    Write-Host "  [OK] Frontend: HTTP $($r.StatusCode)" -ForegroundColor Green
} catch {
    Write-Host "  [FAIL] Frontend not responding" -ForegroundColor Red
}

try {
    $s = Wait-ForHttp "http://localhost:8501/_stcore/health" -TimeoutSec 40
    Write-Host "  [OK] Streamlit health: $($s.Content)" -ForegroundColor Green
} catch {
    Write-Host "  [FAIL] Streamlit not responding" -ForegroundColor Red
}

Write-Host ""
Write-Host "Open frontend:  http://localhost:3000/midan.html" -ForegroundColor Yellow
Write-Host "Open backend:   http://localhost:8000/health" -ForegroundColor Yellow
Write-Host "Open dashboard: http://localhost:8501" -ForegroundColor Yellow
