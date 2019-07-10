#!/bin/bash

mkdir blender-git
cd blender-git

git clone https://github.com/sobotka/blender.git
cd blender
git checkout blender2.7
git submodule update --init

cd ..
mkdir build
cd build

PYENV=$VIRTUAL_ENV
echo $PYENV

cmake -DCMAKE_INSTALL_PREFIX=$PYENV/lib/python3.6/site-packages \
    -DWITH_PYTHON_INSTALL=OFF \
    -DWITH_PYTHON_MODULE=ON \
    -DPYTHON_ROOT_DIR=$PYENV/bin \
    -DPYTHON_SITE_PACKAGES=$PYENV/lib/python3.6/site-packages \
    -DPYTHON_INCLUDE=/usr/include/python3.6/ \
    -DPYTHON_INCLUDE_DIR=$PYENV/include/python3.6m \
    -DPYTHON_LIBRARY=$PYENV/lib/python3.6/config-3.6m-x86_64-linux-gnu/libpython3.6.so \
    -DPYTHON_VERSION=3.6 \
    -DWITH_INSTALL_PORTABLE=OFF \
    -DWITH_CYCLES_EMBREE=OFF \
    -DWITH_CYCLES=ON \
    -DWITH_CYCLES_DEVICE_CUDA=ON \
    -DWITH_OPENSUBDIV=ON \
    -DWITH_OPENAL=OFF \
    -DWITH_CODEC_AVI=ON \
    -DWITH_MOD_OCEANSIM=ON \
    -DWITH_CODEC_FFMPEG=ON \
    -DWITH_SYSTEM_GLEW=ON \
    -DWITH_FFTW3=ON \
    -DWITH_OPENCOLORIO=ON \
    -DWITH_GAMEENGINE=OFF \
    -DWITH_PLAYER=OFF \
    -DWITH_INTERNATIONAL=OFF \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    ../blender

make -j$(nproc)
make install

#cd release/datafiles/locale
#git checkout blender2.7


