#!/usr/bin/bash
set -euo pipefail

cd ~/mpc
git reset --hard
git pull origin main

echo "Деплой завершён!"