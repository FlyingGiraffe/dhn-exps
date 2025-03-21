DATA_NAME=single_pendulum
#DATA_NAME=double_pendulum

SAVE_DIR=data/${DATA_NAME}/train

python data_gen/main.py \
--seed=0 \
--config=data_gen/configs/${DATA_NAME}.py \
--config.save_dir=${SAVE_DIR}