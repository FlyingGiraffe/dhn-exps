##################################################
# Completion
##################################################

EXP_CLASS=ar

#EXP_NAME=sinpend_kernel2_stride1
#EXP_NAME=sinpend_kernel4_stride2
#EXP_NAME=sinpend_kernel8_stride4

#EXP_NAME=sinpend_baseline_hnn_tf
#EXP_NAME=sinpend_baseline_vanilla_tf
#EXP_NAME=sinpend_baseline_vanilla_resnet_layer1
#EXP_NAME=sinpend_baseline_vanilla_resnet_layer2

#EXP_NAME=doupend_kernel2_stride1
#EXP_NAME=doupend_kernel4_stride2
#EXP_NAME=doupend_kernel8_stride4

#EXP_NAME=doupend_baseline_hnn_tf
#EXP_NAME=doupend_baseline_vanilla_tf
#EXP_NAME=doupend_baseline_vanilla_resnet_layer1
#EXP_NAME=doupend_baseline_vanilla_resnet_layer2


RESULT_DIR=results/${EXP_CLASS}/${EXP_NAME}

rm -rf ${RESULT_DIR}/extract

python main.py \
--config=configs/${EXP_CLASS}/${EXP_NAME}.py \
--mode=extract \
--config.workdir=${RESULT_DIR} \
--config.model.num_embeddings=200 \
--config.logging.num_eval_batches=1 \
--config.data.batch_size=100 \
--config.optim.num_epochs=200 \
--config.optim.lr=1e-2 \
--config.loss.crop_interval=0,16 