#!/bin/bash


benchmark() {
    model=$1
    cmd="python3 perf.py --model $model.onnx --seq 64 128 256 512"

    $cmd > $model.csv

    sudo docker rm perf_container --force

    sudo docker run --name perf_container --mount type=bind,src=$(pwd),dst=/ort mcr.microsoft.com/azureml/onnxruntime:latest-nuphar \
        /bin/sh -c "cd ort && $cmd --nuphar >> $model.csv"

    sudo docker wait perf_container
    sudo docker rm perf_container --force
}

benchmark "BLSTM_i64_h128_l2"
benchmark "LSTM_i256_h1024_l4"

echo

echo
echo "benchmark results (BLSTM_i64_h128_l2):"
cat BLSTM_i64_h128_l2.csv

echo
echo "benchmark results (LSTM_i256_h1024_l4):"
cat LSTM_i256_h1024_l4.csv
