#!/bin/bash
 
str1='Hello Bash'
str2="Hello Bash"
analytic=1
 
if [ "$str1" == "Hello Bash" ]; then
    echo "Strings are equal"
else
    echo "Strings are not equal"
fi

if [ $analytic -eq 1 ] ; then
   echo 'analytic'
fi