#!/usr/bin/bash
set -euo pipefail

cd ~
git pull https://github.com/Nepike/mipt_parallel_computations ./mpc
cd ~/mpc

git reset --hard
git pull origin main

echo "Деплой завершён!"