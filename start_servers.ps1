# MIDAN Server Launcher — run this to start both servers
$wd = "C:\Users\seif alaa\asde-project\Intelligent_Systems_Project"
$py = "C:\Python314\python.exe"

Write-Host "Starting MIDAN servers..." -ForegroundColor Cyan

# Kill anything on 8000 or 3000 first
foreach ($port in @(8000, 3000)) {
    $lines = netstat -ano | Select-String ":$port "
    foreach ($line in $lines) {
        $procId = $line.ToString().Trim() -split '\s+' | Select-Object -Last 1
        if ($procId -match '^\d+$' -and [int]$procId -gt 0) {
            Stop-Process -Id ([int]$procId) -Force -ErrorAction SilentlyContinue
        }
    }
}
Start-Sleep 1

# Start FastAPI backend (port 8000)
$psi1 = New-Object System.Diagnostics.ProcessStartInfo
$psi1.FileName = $py
$psi1.Arguments = "-m uvicorn api:api --host 0.0.0.0 --port 8000"
$psi1.WorkingDirectory = $wd
$psi1.UseShellExecute = $false
$psi1.CreateNoWindow = $true
$api = [System.Diagnostics.Process]::Start($psi1)
Write-Host "  API server PID: $($api.Id) → http://localhost:8000" -ForegroundColor Green

# Start static HTTP server (port 3000)
$psi2 = New-Object System.Diagnostics.ProcessStartInfo
$psi2.FileName = $py
$psi2.Arguments = "-m http.server 3000"
$psi2.WorkingDirectory = $wd
$psi2.UseShellExecute = $false
$psi2.CreateNoWindow = $true
$http = [System.Diagnostics.Process]::Start($psi2)
Write-Host "  HTTP server PID: $($http.Id) → http://localhost:3000" -ForegroundColor Green

Start-Sleep 5

# Verify
try {
    $h = Invoke-RestMethod "http://localhost:8000/health" -TimeoutSec 5
    Write-Host "  [OK] API health: models_loaded=$($h.models_loaded)" -ForegroundColor Green
} catch {
    Write-Host "  [FAIL] API not responding" -ForegroundColor Red
}

try {
    $r = Invoke-WebRequest "http://localhost:3000/midan.html" -UseBasicParsing -TimeoutSec 5
    Write-Host "  [OK] Frontend: HTTP $($r.StatusCode)" -ForegroundColor Green
} catch {
    Write-Host "  [FAIL] Frontend not responding" -ForegroundColor Red
}

Write-Host ""
Write-Host "Open: http://localhost:3000/midan.html" -ForegroundColor Yellow
