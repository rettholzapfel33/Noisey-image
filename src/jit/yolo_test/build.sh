cd build
cmake -DCMAKE_PREFIX_PATH=/home/vijay/Documents/devmk4/Noisey-image/jupyter_files/libtorch .. # change prefix to where your libtorch is located
cmake --build . --config Release
