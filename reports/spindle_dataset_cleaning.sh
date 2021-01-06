#!/bin/bash
# (i) delete lines that contain any alphabetical character - e.g. ["table","insert",...], slashes and asterixes, eg. /*, (ii) remove "),", (iii) remove "(", (iv) remove ");", (v) remove empty lines, (vi) insert header into 1st line
# -i : inplace ; -e : expression
INPUT_FILE=$1
OUTPUT_FILE="${INPUT_FILE%%.*}"".csv"
sed -e "/[A-Za-z\/\*]/d" -e "s/),//g; s/(//g; s/);//g" $INPUT_FILE > $OUTPUT_FILE
sed -i -e "/^[[:space:]]*$/d" $OUTPUT_FILE
sed -i -e "1s;^;DriverId,TagId,TS,Value\n;" $OUTPUT_FILE
echo "done"
