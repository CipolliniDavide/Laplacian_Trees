#!/bin/bash

for f in *.hist
do
   b=$(basename "$f" ".hist")
   fgrep -- '->' "$f" | sed 's/   / /' | sed 's/ -/~-/' \
                      | sed 's/ / N/g' | sed 's/~/ /' > ./"$b".tree
   echo "see ./${b}.tree"
done
