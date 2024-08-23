# 2D-Ptr
Source code for paper "2D-Ptr: 2D Array Pointer Network for Solving the Heterogeneous Capacitated Vehicle Routing Problem"

## Dependencies

- Python>=3.8
- NumPy
- SciPy
- [PyTorch](http://pytorch.org/)>=1.12.1
- tqdm
- [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)

## Quick start

The implementation of the 2D-Ptr model is mainly in the file `./nets/attention_model.py`

For testing HCVRP instances with 60 customers and 5 vehicles (V5-U60) and using pre-trained model:

```shell
# greedy
python eval.py data/hcvrp/hcvrp_v5_60_seed24610.pkl --model outputs/hcvrp_v5_60 --obj min-max --decode_strategy greedy --eval_batch_size 1
# sample1280
python eval.py data/hcvrp/hcvrp_v5_60_seed24610.pkl --model outputs/hcvrp_v5_60 --obj min-max --decode_strategy sample --width 1280 --eval_batch_size 1
# sample12800
python eval.py data/hcvrp/hcvrp_v5_60_seed24610.pkl --model outputs/hcvrp_v5_60 --obj min-max --decode_strategy sample --width 12800 --eval_batch_size 1
```

Since AAMAS limits the submission file size within 25Mb, we can only provide the pre-trained model on V5-U60 to avoid exceeding the limit.

**PS: All pre-trained models have been uploaded!**

## Usage

### Generating data

We have provided all the well-generated test datasets in `./data`, and you can also generate each test set by:

```shell
python generate_data.py --dataset_size 1280 --veh_num 3 --graph_size 40
```

- The `--graph_size`  and `--veh_num`  represent the number of customers , vehicles and generated instances, respectively.

- The  default random seed is 24610, and you can change it in `./generate_data.py`.
- The test set will be stored in `./data/hcvrp/`

### Training

For training HCVRP instances with 40 customers and 3 vehicles (V3-U40):

```shell
python run.py --graph_size 40 --veh_num 3 --baseline rollout --run_name hcvrp_v3_40_rollout --obj min-max
```

- `--run_name` will be automatically appended with a timestamp, as the unique subpath for logs and checkpoints.
- The log based on Tensorboard will be stored in `./log/`, and the checkpoint (or the well-trained model) will be stored in `./outputs/`
- `--obj` represents the objective function, supporting `min-max` and `min-sum`

By default, training will happen on all available GPUs.   Change the code in `./run.py` to only use specific GPUs:

```python
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run(get_options())
```

### Evaluation

you can test a well-trained model on HCVRP instances with any problem size:

```shell
# greedy
python eval.py data/hcvrp/hcvrp_v3_40_seed24610.pkl --model outputs/hcvrp_v3_40 --obj min-max --decode_strategy greedy --eval_batch_size 1
# sample1280
python eval.py data/hcvrp/hcvrp_v3_40_seed24610.pkl --model outputs/hcvrp_v3_40 --obj min-max --decode_strategy sample --width 1280 --eval_batch_size 1
# sample12800
python eval.py data/hcvrp/hcvrp_v3_40_seed24610.pkl --model outputs/hcvrp_v3_40 --obj min-max --decode_strategy sample --width 12800 --eval_batch_size 1
```

- The `--model`  represents the directory where the used model is located. 
- The `$filename$.pkl` represents the test set. 
- The `--width` represents sampling number, which is only available when `--decode_strategy` is `sample`.
- The `--eval_batch_size` is set to 1 for serial evaluation.





