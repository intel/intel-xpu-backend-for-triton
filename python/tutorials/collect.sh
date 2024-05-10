#!/bin/bash

# export env variables
source env.sh

# store result
rm -rf result.csv
echo "M/N/K,oneDNN(peak),Triton(peak)" | tee result.csv

for ((i=1; i<=16; i++))
do
    shape_size=$((256 * $i))
    echo "shape size: $shape_size"
    
    rm -rf log.txt

    # update shape size in driver.py and 09-experimental-block-pointer.py
    sed -i "s/for i in \[[0-9]*\]\]/for i in \[$i\]\]/g" 09-experimental-block-pointer.py
    sed -i "s/float M = 256 \* [0-9]*;/float M = 256 \* $i;/g" ../../third_party/intel/backend/driver.py


    python 09-experimental-block-pointer.py 2>&1 | tee log.txt

    oneDNN=`cat log.txt | tail -n 1 | awk '{print $5}'`
    Triton=`cat log.txt | tail -n 1 | awk '{print $6}'`


    oneDNN1=`grep "oneDNN Peak TFlops" log.txt | awk '{print $NF}'`
    Triton1=`grep "Triton Peak TFlops" log.txt | awk '{print $NF}' | awk 'BEGIN{max=0} {if ($1>max) max=$1} END{print max}'`

    echo $shape_size,$oneDNN1,$Triton1 | tee -a result.csv
done

cat result.csv