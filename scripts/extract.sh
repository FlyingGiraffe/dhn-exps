##################################################
# Representation Learning
##################################################

#EXP_CLASS=repn

#EXP_NAME=doupend_kernel2_stride1
#EXP_NAME=doupend_kernel4_stride2
#EXP_NAME=doupend_kernel8_stride4

#EXP_NAME=doupend_kernel4_stride1
#EXP_NAME=doupend_kernel4_stride3

#EXP_NAME=doupend_kernel8_stride1
#EXP_NAME=doupend_kernel8_stride2
#EXP_NAME=doupend_kernel8_stride3
#EXP_NAME=doupend_kernel8_stride5
#EXP_NAME=doupend_kernel8_stride6
#EXP_NAME=doupend_kernel8_stride7

#EXP_NAME=doupend_baseline_hnn_tf
#EXP_NAME=doupend_baseline_vanilla_tf
#EXP_NAME=doupend_baseline_vanilla_resnet_layer1
#EXP_NAME=doupend_baseline_vanilla_resnet_layer2

##################################################
# Super-Resolution
##################################################

#EXP_CLASS=superres

#EXP_NAME=sinpend_4x
#EXP_NAME=sinpend_baseline_cnn_4x

#EXP_NAME=doupend_4x
#EXP_NAME=doupend_baseline_cnn_4x


RESULT_DIR=results/${EXP_CLASS}/${EXP_NAME}

rm -rf ${RESULT_DIR}/extract

python main.py \
--config=configs/${EXP_CLASS}/${EXP_NAME}.py \
--mode=extract \
--config.workdir=${RESULT_DIR} \
--config.model.num_embeddings=200 \
--config.logging.num_eval_batches=1000000 \
--config.data.batch_size=100 \
--config.optim.num_epochs=1000 \
--config.optim.lr=1e-2