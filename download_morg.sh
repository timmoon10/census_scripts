#!/bin/bash
set -x

# Paths
WORK_DIR=$(dirname $(realpath $0))
DATA_DIR=${WORK_DIR}/data/morg
mkdir -p ${DATA_DIR}

# Download MORG data
pushd ${DATA_DIR}
YEARS="79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23"
for YEAR in ${YEARS}; do
    wget https://data.nber.org/morg/annual/morg${YEAR}.dta
done
popd
