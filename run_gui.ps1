$root = $PSScriptRoot

Write-Host ""
Write-Host "=== HVAC Commissioning GUI ===" -ForegroundColor Cyan

# -- Check for npm -------------------------------------------------------------------
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Node.js / npm not found." -ForegroundColor Red
    Write-Host "       Install with:  winget install OpenJS.NodeJS.LTS" -ForegroundColor Yellow
    exit 1
}

# -- Backend dependencies ------------------------------------------------------------
Write-Host "Checking backend dependencies..." -ForegroundColor Gray
pip install -q -r "$root\gui\backend\requirements.txt"

# -- Frontend dependencies -----------------------------------------------------------
if (-not (Test-Path "$root\gui\frontend\node_modules")) {
    Write-Host "Installing frontend dependencies (first run)..." -ForegroundColor Yellow
    Push-Location "$root\gui\frontend"; npm install; Pop-Location
}

# -- Start backend in the background (no new window) --------------------------------
Write-Host "Starting backend on http://localhost:8000 ..." -ForegroundColor Green
$backend = Start-Process python `
    -ArgumentList "-m", "uvicorn", "main:app", "--port", "8000" `
    -WorkingDirectory "$root\gui\backend" `
    -WindowStyle Hidden `
    -PassThru

# -- Open browser after frontend is ready --------------------------------------------
Start-Job { Start-Sleep 5; Start-Process "http://localhost:5173" } | Out-Null

# -- Run frontend in THIS terminal (Ctrl+C stops everything) -------------------------
Write-Host "Frontend on http://localhost:5173  --  press Ctrl+C to stop" -ForegroundColor Cyan
Write-Host ""
try {
    Push-Location "$root\gui\frontend"
    npm run dev
} finally {
    Pop-Location
    Write-Host ""
    Write-Host "Stopping backend (PID $($backend.Id))..." -ForegroundColor Yellow
    Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    Write-Host "Done." -ForegroundColor Gray
}
