#!/bin/bash

#### -------------------------------------------------------------------
#### build docker image
#### -------------------------------------------------------------------
echo 'Building Dockerfile with image name parl-starcraft2:1.0'
docker build -t parl-starcraft2:1.0 .
