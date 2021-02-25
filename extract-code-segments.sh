# EXTRACTS THE RELEVANT CODE-PARTS FROM A LESSON FILE INTO A PYTHON FILE
# RUN WITH BASH OR ZSH, NOT SH!

input_file=$1
# echo ${input_file/md/py}
sed -n -E '/#@tab (pytorch|all)/,/```/p' $input_file |\
    sed -E -e 's/#@tab (pytorch|all)//' -e 's/```//' >\
    ${input_file/md/py}
