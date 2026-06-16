#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ANSI colours
cyan='\033[36m'; red='\033[31m'; yellow='\033[33m'; green='\033[32m'; gray='\033[90m'; reset='\033[0m'

echo ""
echo -e "${cyan}=== HVAC Commissioning GUI ===${reset}"

# -- Check for npm -------------------------------------------------------------------
if ! command -v npm >/dev/null 2>&1; then
    echo -e "${red}ERROR: Node.js / npm not found.${reset}"
    echo -e "${yellow}       Install with:  sudo apt install nodejs npm  (or use nvm)${reset}"
    exit 1
fi

# -- Pick a Python interpreter -------------------------------------------------------
PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
    echo -e "${red}ERROR: Python not found.${reset}"
    echo -e "${yellow}       Install with:  sudo apt install python3 python3-venv${reset}"
    exit 1
fi

# -- Backend dependencies (in a virtualenv, per PEP 668) -----------------------------
venv="$root/gui/backend/.venv"
if [ ! -d "$venv" ]; then
    echo -e "${yellow}Creating Python virtual environment (first run)...${reset}"
    if ! "$PY" -m venv "$venv"; then
        echo -e "${red}ERROR: Could not create venv.${reset}"
        echo -e "${yellow}       Install with:  sudo apt install python3-venv${reset}"
        exit 1
    fi
fi
PY="$venv/bin/python"

echo -e "${gray}Checking backend dependencies...${reset}"
"$PY" -m pip install -q -r "$root/gui/backend/requirements.txt"

# -- Frontend dependencies -----------------------------------------------------------
if [ ! -d "$root/gui/frontend/node_modules" ]; then
    echo -e "${yellow}Installing frontend dependencies (first run)...${reset}"
    (cd "$root/gui/frontend" && npm install)
fi

# -- Start backend in the background -------------------------------------------------
echo -e "${green}Starting backend on http://localhost:8000 ...${reset}"
(cd "$root/gui/backend" && "$PY" -m uvicorn main:app --port 8000) &
backend_pid=$!

# -- Stop backend on exit (Ctrl+C stops everything) ---------------------------------
cleanup() {
    echo ""
    echo -e "${yellow}Stopping backend (PID $backend_pid)...${reset}"
    kill "$backend_pid" 2>/dev/null || true
    echo -e "${gray}Done.${reset}"
}
trap cleanup EXIT INT TERM

# -- Open browser after frontend is ready --------------------------------------------
( sleep 5; xdg-open "http://localhost:5173" >/dev/null 2>&1 || true ) &

# -- Run frontend in THIS terminal ---------------------------------------------------
echo -e "${cyan}Frontend on http://localhost:5173  --  press Ctrl+C to stop${reset}"
echo ""
cd "$root/gui/frontend"
npm run dev
