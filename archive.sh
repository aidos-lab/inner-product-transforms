#! /bin/bash 

d=$(date '+%Y-%m-%d-%H-%M')

echo "$1.zip"
zip -r "./archive/$1-$d.zip" my_test 
