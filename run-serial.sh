#!/bin/bash

MAX_STEPS=(2000 4000 8000 16000 32000 64000 128000)
LOG="results.log"
RUNS=10
SCALE=6
USE_SRUN=0

run_demo() {
    if (( USE_SRUN )); then
        srun -n 1 -c 1 demo/demo "$@" 2>/dev/null
    else
        demo/demo "$@" 2>/dev/null
    fi
}

run_serial() {
    local SCENARIO="$1"

    echo "Running serial with scenario: $SCENARIO..."
    echo "##########################################################" >> "$LOG"
    echo "(SCENARIO, MAX_STEPS) -> TIME[s]" >> "$LOG"

    for max_steps in "${MAX_STEPS[@]}"; do
        sum=0
        for ((i=1; i<=RUNS; i++)); do
            value=$(run_demo --seq --timing --max-steps="$max_steps" "$SCENARIO" \
                        | grep "^Target time" \
                        | cut -d' ' -f3)
            sum=$(bc <<< "scale=$SCALE; $sum + $value")
        done
        # also convert from miliseconds to seconds
        avg=$(bc <<< "scale=$SCALE; $sum / $RUNS / 1000")
        printf "(%s, %d) -> %.*f\n" \
            "$SCENARIO" "$max_steps" "$SCALE" "$avg" >> "$LOG"
    done

    echo "Running $SCENARIO complete"
}

if [[ "$1" == "--srun" ]]; then
    echo "Using srun..."
    USE_SRUN=1
fi
[ -f "$LOG" ] && mv -v "$LOG" "$LOG.old"

echo "Running serial benchmarks..."

run_serial "hugeScenario.xml"
run_serial "scenario_box.xml"
run_serial "scenario.xml"

echo "Running serial benchmarks complete"
