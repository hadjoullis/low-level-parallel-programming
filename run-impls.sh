#!/bin/bash

MAX_STEPS=(200 800 3200 12800 51200 204800)
LOG="results.log"
RUNS=10
SCALE=3
USE_SRUN=0

run_demo() {
  local threads=$1
  shift

  if ((USE_SRUN)); then
    srun -n 1 -c "$threads" demo/demo "$@" 2>/dev/null
  else
    demo/demo "$@" 2>/dev/null
  fi
}

run_serial() {
  local SCENARIO="$1"

  echo "Running serial with scenario: $SCENARIO..."
  echo "##########################################################" >>"$LOG"
  echo "(SERIAL, SCENARIO, MAX_STEPS) -> (TIME[s], SPEEDUP)" >>"$LOG"

  for max_steps in "${MAX_STEPS[@]}"; do
    sum_time=0
    sum_speedup=0
    for ((i = 1; i <= RUNS; i++)); do
      ret=$(run_demo 1 --seq --timing --max-steps="$max_steps" "$SCENARIO")

      value_time=$(grep '^Target time' <<<"$ret" | cut -d' ' -f3)
      value_speedup=$(grep '^Speedup' <<<"$ret" | cut -d' ' -f2)

      sum_time=$(bc <<<"scale=$SCALE; $sum_time + $value_time")
      sum_speedup=$(bc <<<"scale=$SCALE; $sum_speedup + $value_speedup")
    done
    # also convert from miliseconds to seconds
    avg_time=$(bc <<<"scale=$SCALE; $sum_time / $RUNS / 1000")
    avg_speedup=$(bc <<<"scale=$SCALE; $sum_speedup / $RUNS")
    printf "(serial, %s, %d) -> (%.${SCALE}f, %.${SCALE}f)\n" \
      "$SCENARIO" "$max_steps" "$avg_time" "$avg_speedup" >>"$LOG"
  done

  echo "Running $SCENARIO complete"
}

run_omp() {
  local SCENARIO="$1"

  echo "Running omp with scenario: $SCENARIO..."
  echo "##########################################################" >>"$LOG"
  echo "(OMP, SCENARIO, MAX_STEPS) -> (TIME[s], SPEEDUP)" >>"$LOG"

  for max_steps in "${MAX_STEPS[@]}"; do
    sum_time=0
    sum_speedup=0
    for ((i = 1; i <= RUNS; i++)); do
      ret=$(OMP_SCHEDULE="dynamic,1" OMP_NUM_THREADS="16" \
        run_demo "$threads" --omp --timing --max-steps="$max_steps" "$SCENARIO")

      value_time=$(grep '^Target time' <<<"$ret" | cut -d' ' -f3)
      value_speedup=$(grep '^Speedup' <<<"$ret" | cut -d' ' -f2)

      sum_time=$(bc <<<"scale=$SCALE; $sum_time + $value_time")
      sum_speedup=$(bc <<<"scale=$SCALE; $sum_speedup + $value_speedup")
    done
    # also convert from miliseconds to seconds
    avg_time=$(bc <<<"scale=$SCALE; $sum_time / $RUNS / 1000")
    avg_speedup=$(bc <<<"scale=$SCALE; $sum_speedup / $RUNS")
    printf "(omp, %s, %d) -> (%.${SCALE}f, %.${SCALE}f)\n" \
      "$SCENARIO" "$max_steps" "$avg_time" "$avg_speedup" >>"$LOG"
  done

  echo "Running $SCENARIO complete"
}

run_pthread() {
  local SCENARIO="$1"

  echo "Running pthread with scenario: $SCENARIO..."
  echo "##########################################################" >>"$LOG"
  echo "(PTHREAD, SCENARIO, MAX_STEPS) -> (TIME[s], SPEEDUP)" >>"$LOG"

  for max_steps in "${MAX_STEPS[@]}"; do
    sum_time=0
    sum_speedup=0
    for ((i = 1; i <= RUNS; i++)); do
      ret=$(PTHREAD_NUM_THREADS=4 \
        run_demo 4 --pthread --timing --max-steps="$max_steps" "$SCENARIO")

      value_time=$(grep '^Target time' <<<"$ret" | cut -d' ' -f3)
      value_speedup=$(grep '^Speedup' <<<"$ret" | cut -d' ' -f2)

      sum_time=$(bc <<<"scale=$SCALE; $sum_time + $value_time")
      sum_speedup=$(bc <<<"scale=$SCALE; $sum_speedup + $value_speedup")
    done
    # also convert from miliseconds to seconds
    avg_time=$(bc <<<"scale=$SCALE; $sum_time / $RUNS / 1000")
    avg_speedup=$(bc <<<"scale=$SCALE; $sum_speedup / $RUNS")
    printf "(pthread, %s, %d) -> (%.${SCALE}f, %.${SCALE}f)\n" \
      "$SCENARIO" "$max_steps" "$avg_time" "$avg_speedup" >>"$LOG"
  done

  echo "Running $SCENARIO complete"
}

run_simd() {
  local SCENARIO="$1"

  echo "Running simd with scenario: $SCENARIO..."
  echo "##########################################################" >>"$LOG"
  echo "(SIMD, SCENARIO, MAX_STEPS) -> (TIME[s], SPEEDUP)" >>"$LOG"

  for max_steps in "${MAX_STEPS[@]}"; do
    sum_time=0
    sum_speedup=0
    for ((i = 1; i <= RUNS; i++)); do
      ret=$(run_demo 1 --simd --timing --max-steps="$max_steps" "$SCENARIO")

      value_time=$(grep '^Target time' <<<"$ret" | cut -d' ' -f3)
      value_speedup=$(grep '^Speedup' <<<"$ret" | cut -d' ' -f2)

      sum_time=$(bc <<<"scale=$SCALE; $sum_time + $value_time")
      sum_speedup=$(bc <<<"scale=$SCALE; $sum_speedup + $value_speedup")
    done
    # also convert from miliseconds to seconds
    avg_time=$(bc <<<"scale=$SCALE; $sum_time / $RUNS / 1000")
    avg_speedup=$(bc <<<"scale=$SCALE; $sum_speedup / $RUNS")
    printf "(simd, %s, %d) -> (%.${SCALE}f, %.${SCALE}f)\n" \
      "$SCENARIO" "$max_steps" "$avg_time" "$avg_speedup" >>"$LOG"
  done

  echo "Running $SCENARIO complete"
}



run_cuda() {
  local SCENARIO="$1"

  echo "Running cuda with scenario: $SCENARIO..."
  echo "##########################################################" >>"$LOG"
  echo "(CUDA, SCENARIO, MAX_STEPS) -> (TIME[s], SPEEDUP)" >>"$LOG"

  for max_steps in "${MAX_STEPS[@]}"; do
    sum_time=0
    sum_speedup=0
    for ((i = 1; i <= RUNS; i++)); do
      ret=$(run_demo 1 --cuda --timing --max-steps="$max_steps" "$SCENARIO")

      value_time=$(grep '^Target time' <<<"$ret" | cut -d' ' -f3)
      value_speedup=$(grep '^Speedup' <<<"$ret" | cut -d' ' -f2)

      sum_time=$(bc <<<"scale=$SCALE; $sum_time + $value_time")
      sum_speedup=$(bc <<<"scale=$SCALE; $sum_speedup + $value_speedup")
    done
    # also convert from miliseconds to seconds
    avg_time=$(bc <<<"scale=$SCALE; $sum_time / $RUNS / 1000")
    avg_speedup=$(bc <<<"scale=$SCALE; $sum_speedup / $RUNS")
    printf "(cuda, %s, %d) -> (%.${SCALE}f, %.${SCALE}f)\n" \
      "$SCENARIO" "$max_steps" "$avg_time" "$avg_speedup" >>"$LOG"
  done



[ -f "$LOG" ] && mv -v "$LOG" "$LOG.old"
echo "Writing logs to '"$LOG"'..."

if [[ "$1" == "--srun" ]]; then
  echo "Using srun..."
  echo "########################## SRUN ##########################" >>"$LOG"
  USE_SRUN=1
fi

########### serial ###########
echo "Running serial benchmarks..."

run_serial "hugeScenario.xml"
run_serial "scenario_box.xml"
run_serial "scenario.xml"

echo "Running serial benchmarks complete"

########### omp ###########
echo "Running omp benchmarks..."

run_omp "hugeScenario.xml"
run_omp "scenario_box.xml"
run_omp "scenario.xml"

echo "Running omp benchmarks complete"

########### pthread ###########
echo "Running pthread benchmarks..."

run_pthread "hugeScenario.xml"
run_pthread "scenario_box.xml"
run_pthread "scenario.xml"

echo "Running pthread benchmarks complete"

########### simd ###########
echo "Running simd benchmarks..."

run_simd "hugeScenario.xml"
run_simd "scenario_box.xml"
run_simd "scenario.xml"

echo "Running simd benchmarks complete"

########### cuda ###########
echo "Running cuda benchmarks..."

run_cuda "hugeScenario.xml"
run_cuda "scenario_box.xml"
run_cuda "scenario.xml"

echo "Running cuda benchmarks complete"
