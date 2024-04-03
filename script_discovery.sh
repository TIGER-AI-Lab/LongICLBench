model=$1
cuda=$2
test_number=$3
api_key=$4

export CUDA_VISIBLE_DEVICES=${cuda}
python3 discovery_infer.py --round 1 -m "${model}" --test_number "${test_number}" --api_key "${api_key}"
python3 discovery_infer.py --round 2 -m "${model}" --test_number "${test_number}" --api_key "${api_key}"
python3 discovery_infer.py --round 3 -m "${model}" --test_number "${test_number}" --api_key "${api_key}"
python3 discovery_infer.py --round 4 -m "${model}" --test_number "${test_number}" --api_key "${api_key}"
python3 discovery_infer.py --round 5 -m "${model}" --test_number "${test_number}" --api_key "${api_key}"
