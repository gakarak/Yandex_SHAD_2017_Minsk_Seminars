#!/bin/bash

finp="idx.txt"
finpShuf="${finp}-shuf.txt"

foutTrain="${finp}-train.txt"
foutVal="${finp}-val.txt"

strHeader=`head -n1 ${finp}`

#############################
cat $finp | grep -v 'path' | shuf > $finpShuf
numFn=`cat $finpShuf | wc -l`
pVal=10

((numVal=numFn*pVal/100))
((numTrain=numFn-numVal))

echo "train/val/tot = ${numTrain}/${numVal}/${numFn}"

#############################
echo "${strHeader}" > $foutTrain
cat $finpShuf | head -n $numTrain >> $foutTrain

echo "${strHeader}" > $foutVal
cat $finpShuf | tail -n $numVal   >> $foutVal
