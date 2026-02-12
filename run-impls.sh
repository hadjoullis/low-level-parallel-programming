#!/bin/bash

MAX_STEPS=(200 800 3200 12800 51200 204800)
LOG="results.log"
RUNS=10
SCALE=3
USE_SRUN=0

run_demo() {
    local threads=$1
    shift

    if (( USE_SRUN )); then
        srun -n 1 -c "$threads" demo/demo "$@" 2>/dev/null
    else
        demo/demo "$@" 2>/dev/null
    fi
}

run_serial() {
    local SCENARIO="$1"

    echo "Running serial with scenario: $SCENARIO..."
    echo "##########################################################" >> "$LOG"
    echo "(serial, SCENARIO, MAX_STEPS) -> (TIME[s], SPEEDUP)" >> "$LOG"

    for max_steps in "${MAX_STEPS[@]}"; do
        sum_time=0
        sum_speedup=0
        for ((i=1; i<=RUNS; i++)); do
            ret=$(run_demo 1 --seq --timing --max-steps="$max_steps" "$SCENARIO")

            value_time=$(grep '^Target time' <<< "$ret" | cut -d' ' -f3)
            value_speedup=$(grep '^Speedup'  <<< "$ret" | cut -d' ' -f2)

            sum_time=$(bc <<< "scale=$SCALE; $sum_time + $value_time")
            sum_speedup=$(bc <<< "scale=$SCALE; $sum_speedup + $value_speedup")
        done
        # also convert from miliseconds to seconds
        avg_time=$(bc <<< "scale=$SCALE; $sum_time / $RUNS / 1000")
        avg_speedup=$(bc <<< "scale=$SCALE; $sum_speedup / $RUNS")
        printf "(serial, %s, %d) -> (%.${SCALE}f, %.${SCALE}f)\n" \
            "$SCENARIO" "$max_steps" "$avg_time" "$avg_speedup" >> "$LOG"
    done

    echo "Running $SCENARIO complete"
}

[ -f "$LOG" ] && mv -v "$LOG" "$LOG.old"
echo "Writing logs to '"$LOG"'..."

if [[ "$1" == "--srun" ]]; then
    echo "Using srun..."
    echo "########################## SRUN ##########################" >> "$LOG"
    USE_SRUN=1
fi

echo "Running serial benchmarks..."

run_serial "hugeScenario.xml"
run_serial "scenario_box.xml"
run_serial "scenario.xml"

echo "Running serial benchmarks complete"
