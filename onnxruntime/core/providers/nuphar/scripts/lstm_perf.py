import argparse
import sys
import os

import numpy as np

import onnxruntime

from timeit import default_timer as timer


def top_n_avg(per_iter_cost, n):
    # following the perf test methodology in [timeit](https://docs.python.org/3/library/timeit.html#timeit.Timer.repeat)
    per_iter_cost.sort()
    return sum(per_iter_cost[:n]) * 1000 / n


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="LSTMx4.onnx")
    parser.add_argument('--repeat', type=int, default=20)
    parser.add_argument('--seq', type=int, nargs='+')
    parser.add_argument('--top_n', type=int, default=5)
    parser.add_argument('--nuphar', action='store_true')
    parser.add_argument('--noquant', action='store_true')

    return parser.parse_args()


def set_num_threads(num_threads):
    if num_threads:
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
    else:
        del os.environ['OMP_NUM_THREADS']


def run(args):
    model = args.model

    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = 1

    sess = onnxruntime.InferenceSession(model, providers=['CPUExecutionProvider'], sess_options=sess_options)

    if args.nuphar:
        assert 'NupharExecutionProvider' in onnxruntime.get_available_providers()
        quant_model = args.model.replace(".onnx", "_scan_int8.onnx")

        set_num_threads(1)
        sess_quant = onnxruntime.InferenceSession(quant_model, providers=['NupharExecutionProvider'])
    else:
        quant_model = args.model.replace(".onnx", "_quant.onnx")
        sess_quant = onnxruntime.InferenceSession(quant_model, providers=['CPUExecutionProvider'], sess_options=sess_options)

    input_dim = sess.get_inputs()[0].shape[2]

    result = []
    for seq in args.seq:
        per_iter_cost = []
        for i in range(args.repeat):
            input_data = np.random.rand(seq, 1, input_dim).astype(np.float32)
            lstm_feed = {sess.get_inputs()[0].name:input_data}

            iter_start = timer()
            sess.run([], lstm_feed)
            end = timer()
            per_iter_cost.append(end - iter_start)

        avg_rnn = top_n_avg(per_iter_cost, args.top_n)
        print(f'perf_rnn {model} seq {seq}: run for {args.repeat} iterations, top {args.top_n} avg {avg_rnn:.3f} ms', file=sys.stderr)
        result.append(avg_rnn)

        try:
            per_iter_cost = []
            for i in range(args.repeat):
                input_data = np.random.rand(seq, 1, input_dim).astype(np.float32)
                lstm_feed = {sess_quant.get_inputs()[0].name:input_data}

                iter_start = timer()
                sess_quant.run([], lstm_feed)
                end = timer()
                per_iter_cost.append(end - iter_start)

            avg_rnn = top_n_avg(per_iter_cost, args.top_n)
            result.append(avg_rnn)
        except Exception:
            result.append("N/A")

        print(f'perf_rnn {quant_model} seq {seq}: run for {args.repeat} iterations, top {args.top_n} avg {avg_rnn:.3f} ms', file=sys.stderr)

    print(result)

if __name__ == "__main__":
    run(parse_arguments())
