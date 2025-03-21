DATA_NAME=single_pendulum
#DATA_NAME=double_pendulum

SAVE_DIR=data/${DATA_NAME}/test

python data_gen/main.py \
--seed=1 \
--config=data_gen/configs/${DATA_NAME}.py \
--config.save_dir=${SAVE_DIR} \
--config.num_data=200