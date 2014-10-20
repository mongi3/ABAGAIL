@echo off
REM NOTE: assumes ABAGAIL.jar has already been created (see ABAGAIL wiki)

set CLASSPATH=..\ABAGAIL.jar;c:\jython\jython.jar;%CLASSPATH%
mkdir data\plot logs image 2> NUL

echo "four peaks"
java org.python.util.jython fourpeaks.py

echo "count ones"
java org.python.util.jython countones.py

echo "continuous peaks"
java org.python.util.jython continuouspeaks.py

echo "Running knapsack"
java org.python.util.jython knapsack.py

