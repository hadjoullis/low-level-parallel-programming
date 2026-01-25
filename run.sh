#!/bin/bash

SCENARIO="hugeScenario.xml"
LOG="results.log"
THREADS=(1 2 4 8 16)
RUNS=10
SCALE=6
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

run_omp_static() {
    echo "Running OMP static..."
    echo "##########################################################" >> "$LOG"
    echo "(OMP_STATIC, THREADS) -> SPEEDUP" >> "$LOG"
    for threads in "${THREADS[@]}"; do
        sum=0
        for ((i=1; i<=RUNS; i++)); do
            value=$(OMP_SCHEDULE=static OMP_NUM_THREADS="$threads" \
                        run_demo "$threads" --omp --timing "$SCENARIO" \
                        | grep '^Speedup' \
                        | cut -d' ' -f2)
            sum=$(bc <<< "scale=$SCALE; $sum + $value")
        done
        avg=$(bc <<< "scale=$SCALE; $sum / $RUNS")
        printf "(OMP_STATIC, %d) -> %.*f\n" "$threads" "$SCALE" "$avg" >> "$LOG"
    done

    echo "Running OMP static complete"
}

run_omp_dynamic() {
    echo "Running OMP dynamic..."

    echo "##########################################################" >> "$LOG"
    echo "(OMP_DYNAMIC, THREADS, CHUNK_SZ) -> SPEEDUP" >> "$LOG"
    local CHUNK_SZ=(1 8 16 64 256)
    for chunk_sz in "${CHUNK_SZ[@]}"; do
        for threads in "${THREADS[@]}"; do
            sum=0
            for ((i=1; i<=RUNS; i++)); do
                value=$(OMP_SCHEDULE="dynamic,$chunk_sz" OMP_NUM_THREADS="$threads" \
                            run_demo "$threads" --omp --timing "$SCENARIO" \
                            | grep '^Speedup' \
                            | cut -d' ' -f2)
                sum=$(bc <<< "scale=$SCALE; $sum + $value")
            done
            avg=$(bc <<< "scale=$SCALE; $sum / $RUNS")
            printf "(OMP_DYNAMIC, %d, %d) -> %.*f\n" "$threads" "$chunk_sz" "$SCALE" "$avg" >> "$LOG"
        done
    done

    echo "Running OMP dynamic complete"
}

run_pthread_static() {
    echo "Running PTHREAD static..."

    echo "##########################################################" >> "$LOG"
    echo "(PTHREAD_STATIC, THREADS) -> SPEEDUP" >> "$LOG"
    for threads in "${THREADS[@]}"; do
        sum=0
        for ((i=1; i<=RUNS; i++)); do
            value=$(PTHREAD_NUM_THREADS="$threads" \
                        run_demo "$threads" --pthread --timing "$SCENARIO" \
                        | grep '^Speedup' \
                        | cut -d' ' -f2)
            sum=$(bc <<< "scale=$SCALE; $sum + $value")
        done
        avg=$(bc <<< "scale=$SCALE; $sum / $RUNS")
        printf "(PTHREAD_STATIC, %d) -> %.*f\n" "$threads" "$SCALE" "$avg" >> "$LOG"
    done

    echo "Running PTHREAD static complete"
}

if [[ "$1" == "--srun" ]]; then
    echo "Using srun..."
    USE_SRUN=1
fi
[ -f "$LOG" ] && mv -v "$LOG" "$LOG.old"

echo "Running benchmarks..."

run_omp_static
run_omp_dynamic
run_pthread_static

echo "Running benchmarks complete"
