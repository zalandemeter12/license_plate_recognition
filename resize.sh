#!/bin/bash

for i in *.png; do
    printf "Resize $i\n"
    convert "$i" -resize 200x200 "$i"
done
