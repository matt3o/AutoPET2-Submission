#!/usr/bin/env bash

./build.sh

docker save sw_segmentation | gzip -c > sw_segmentation.tar.gz
