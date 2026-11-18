#!/bin/bash

cd "$(dirname "$0")"
LAYOUT=${LAYOUT:-warehouse}
STEPS=${STEPS:-1200000}
JOBS=${JOBS:-$(( $(nproc) > 2 ? $(nproc) - 1 : 1 ))}
mkdir -p ../logs/$LAYOUT/console

echo "Помещение: $LAYOUT, шагов на прогон: $STEPS, параллельно: $JOBS"
echo "Консольный вывод каждого прогона: logs/$LAYOUT/console/<имя>.txt"
START=$(date +%s)

for algo in mappo ippo; do
  for reward in global individual difference; do
    for seed in 0 1 2; do
      echo "$algo $reward $seed"
    done
  done
done | xargs -P "$JOBS" -L 1 bash -c '
  name="$0_$1_s$2"
  L='"$LAYOUT"'
  if [ -f "../logs/$L/$name/model.pt" ]; then echo "[пропуск] $name уже обучен"; exit 0; fi
  echo "[старт]   $name"
  python ippo_mappo.py --layout "$L" --algo "$0" --reward "$1" --seed "$2" --steps '"$STEPS"' > "../logs/$L/console/$name.txt" 2>&1
  echo "[готово]  $name"; tail -n 2 "../logs/$L/console/$name.txt" | head -n 1
'

echo "Все прогоны завершены за $(( ($(date +%s) - START) / 60 )) мин."
