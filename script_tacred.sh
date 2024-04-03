model=$1
cuda=$2
test_number=$3

export CUDA_VISIBLE_DEVICES=${cuda}
python3 tacred_infer.py --round 1 -m "${model}" --test_number "${test_number}"
python3 tacred_infer.py --round 2 -m "${model}" --test_number "${test_number}"
python3 tacred_infer.py --round 3 -m "${model}" --test_number "${test_number}"
python3 tacred_infer.py --round 4 -m "${model}" --test_number "${test_number}"
python3 tacred_infer.py --round 5 -m "${model}" --test_number "${test_number}"