
#Setup program and data if not done already
if [ ! -f cluster.exe ]; then
    echo -e "Building program"
    make clean
    make
fi

if [ ! -f data/fireNew.vectors ]; then
    echo -e "Extracting data file"
    unzip data/fireNew.zip -d data/

fi

echo -e "\n\n------------Test 1; clusters 50, vectors 5000 --------------\n"
./cluster.exe 50 "data/fireNew.vectors" 5000


echo -e "\n\n------------Test 2; clusters 100, vectors 5000 --------------\n"
./cluster.exe 100 "data/fireNew.vectors" 5000


echo -e "\n\n------------Test 3; clusters 50, vectors 10000 --------------\n"
./cluster.exe 50 "data/fireNew.vectors" 10000


echo -e "\n\n------------Test 4; clusters 100, vectors 10000 --------------\n"
./cluster.exe 100 "data/fireNew.vectors" 10000