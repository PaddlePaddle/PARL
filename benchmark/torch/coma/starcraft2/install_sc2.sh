#!/bin/bash

#### -------------------------------------------------------------------
#### Install SC2 and add the custom maps
#### -------------------------------------------------------------------

if [ -z "$SC2PATH" ]
then
    SC2PATH=`pwd`'/StarCraftII'
else
    SC2PATH=$SC2PATH'/StarCraftII'
fi

export SC2PATH=$SC2PATH
echo 'SC2PATH is set to '$SC2PATH

if [ ! -d $SC2PATH ]; then
        echo 'StarCraftII is not installed. Installing now ...';
        wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.6.2.69232.zip
        unzip -P iagreetotheeula SC2.4.6.2.69232.zip
        rm -rf SC2.4.6.2.69232.zip
        echo 'Finished installing StarCraftII'
else
        echo 'StarCraftII is already installed.'
fi

echo 'Adding SMAC maps.'
MAP_DIR="$SC2PATH/Maps/"
echo 'MAP_DIR is set to '$MAP_DIR
mkdir -p $MAP_DIR

cd ..
wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
unzip SMAC_Maps.zip
mv SMAC_Maps $MAP_DIR
rm -rf SMAC_Maps.zip

echo 'StarCraft II and SMAC are installed.'

