#!/bin/bash
rm -rf log.txt
python 09-experimental-block-pointer.py 2>&1 | tee log.txt


oneDNN=`cat log.txt | tail -n 1 | awk '{print $5}'`
Triton=`cat log.txt | tail -n 1 | awk '{print $6}'`


oneDNN1=`grep "oneDNN Peak TFlops" log.txt | awk '{print $NF}'`
Triton1=`grep "Triton Peak TFlops" log.txt | awk '{print $NF}' | awk 'BEGIN{max=0} {if ($1>max) max=$1} END{print max}'`

echo $oneDNN,$Triton,$oneDNN1,$Triton1 | tee -a data.csv