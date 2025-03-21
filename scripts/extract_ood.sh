##################################################
# Super-Resolution
##################################################

EXP_CLASS=superres

#EXP_NAME=sinpend_4x
#EXP_NAME=sinpend_baseline_cnn_4x

#EXP_NAME=doupend_4x
#EXP_NAME=doupend_baseline_cnn_4x


RESULT_DIR=results/${EXP_CLASS}/${EXP_NAME}

rm -rf ${RESULT_DIR}/extract_ood

python main.py \
--config=configs/${EXP_CLASS}/${EXP_NAME}.py \
--mode=extract \
--config.workdir=${RESULT_DIR} \
--config.model.num_embeddings=200 \
--config.logging.num_eval_batches=1000000 \
--config.data.batch_size=100 \
--config.optim.num_epochs=1000 \
--config.optim.lr=1e-2 \
--config.model.train_step_span=512,1025 \
--work_subdir=extract_ood