#!/bin/bash

set -e

download() {
    wget ftp://ftp.irisa.fr/local/texmex/corpus/$1.tar.gz
    tar xf $1.tar.gz
    rm $1.tar.gz
}

download siftsmall
download sift