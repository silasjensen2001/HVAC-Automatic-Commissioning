$root = $PSScriptRoot

Write-Host ""
Write-Host "=== HVAC Commissioning GUI ===" -ForegroundColor Cyan

# ── Check for npm before doing anything else ─────────────────────────────────
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "ERROR: Node.js / npm not found." -ForegroundColor Red
    Write-Host "       Install it with:  winget install OpenJS.NodeJS.LTS" -ForegroundColor Yellow
    Write-Host "       Then restart this terminal and run the script again." -ForegroundColor Yellow
    exit 1
}

# ── Install backend dependencies if needed ────────────────────────────────────
$pipReqs = "$root\gui\backend\requirements.txt"
Write-Host "Checking backend dependencies..." -ForegroundColor Gray
pip install -q -r $pipReqs

# ── Install frontend dependencies if node_modules is missing ─────────────────
$nodeModules = "$root\gui\frontend\node_modules"
if (-not (Test-Path $nodeModules)) {
    Write-Host "Installing frontend dependencies (first run)..." -ForegroundColor Yellow
    Push-Location "$root\gui\frontend"
    npm install
    Pop-Location
}

# ── Launch backend ────────────────────────────────────────────────────────────
Write-Host "Starting backend on http://localhost:8000 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList `
    "-NoExit", "-Command", `
    "cd '$root\gui\backend'; python -m uvicorn main:app --reload --port 8000"

# ── Launch frontend ───────────────────────────────────────────────────────────
Write-Host "Starting frontend on http://localhost:5173 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList `
    "-NoExit", "-Command", `
    "cd '$root\gui\frontend'; npm run dev"

# ── Open browser after a short delay ─────────────────────────────────────────
Start-Sleep -Seconds 4
Start-Process "http://localhost:5173"

Write-Host ""
Write-Host "GUI launched. Close the two terminal windows to stop the servers." -ForegroundColor Cyan
