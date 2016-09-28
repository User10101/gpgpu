#!/bin/bash

prog=$1
ofile=$2

for offset in `seq 0 16` 
do
    echo "$offset"
    ./"$prog" 1024 512 "$offset" >> "$ofile" 
done
