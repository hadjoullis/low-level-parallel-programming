#!/bin/bash

IMPLS=("omp" "pthread" "simd" "cuda")
SCENARIO="scenario.xml"
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

run_demo() {
	local name="$(sed  's/-//g' <<<"$1")"
	demo/demo "$@" --export-trace="/tmp/trace-$name.bin" "$SCENARIO" &>/dev/null
}

run_serial() {
	run_demo --seq
}

run_omp() {
	OMP_SCHEDULE="dynamic,1" OMP_NUM_THREADS=16 run_demo --omp
}

run_pthread() {
	PTHREAD_NUM_THREADS=4 run_demo --pthread
}

run_simd() {
	run_demo --simd
}

run_cuda() {
	run_demo --cuda
}

echo "Testing implementations..."

run_serial

for impl in "${IMPLS[@]}"; do
 	target="/tmp/trace-$impl.bin"
	rm -f "$target"
	echo -n "Testing $impl... "
	"run_$impl"

	if cmp -s "/tmp/trace-seq.bin" "$target"; then
		echo -e "${GREEN}SUCCESS${NC}"
	else
		echo -e "${RED}FAILURE${NC}"
	fi
done

echo "Testing implementations complete"
