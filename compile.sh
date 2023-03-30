hipcc -c -o cusage.o cusage.cc
ar rcs libcusage.a cusage.o
rm *.o