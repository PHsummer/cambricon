ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:22.01-tf1-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace
RUN git clone https://github.com/williamleif/GraphSAGE.git && cd GraphSAGE && git checkout a0fdef95dca7b456dab01cb35034717c8b6dd017 && sed -i 's|"time=", "{:.5f}".format(avg_time)|"time=", "{:.5f}".format(avg_time),"rate_avg", "{:>7.2f} nodes/s".format(FLAGS.batch_size/avg_time)|g' ./graphsage/unsupervised_train.py && sed -i 's|"time=", "{:.5f}".format(avg_time)|"time=", "{:.5f}".format(avg_time),"rate_avg", "{:>7.2f} nodes/s".format(FLAGS.batch_size/avg_time)|g' ./graphsage/supervised_train.py && sed -i 's/train(train_data)/print("batch_size=",format(FLAGS.batch_size)),train(train_data)/g' ./graphsage/unsupervised_train.py && sed -i 's|labels = {int(i):l for i, l in labels.iteritems()}|labels = {int(i):l for i, l in labels.items()}|g' ./eval_scripts/ppi_eval.py && sed -i 's|log = MultiOutputClassifier(SGDClassifier(loss="log"), n_jobs=10)|log = MultiOutputClassifier(SGDClassifier(loss="log",random_state=1), n_jobs=10)|g' ./eval_scripts/ppi_eval.py && sed -i '27,30d' ./eval_scripts/ppi_eval.py && sed -i '27s/$/\n    print("F1 score", f1_score(test_labels, log.predict(test_embeds), average="micro"))/g' ./eval_scripts/ppi_eval.py && sed -i '28s/$/\n    print("Random baseline F1 score", f1_score(test_labels, dummy.predict(test_embeds), average="micro"))/g' ./eval_scripts/ppi_eval.py
 
WORKDIR /workspace/GraphSAGE 
RUN pip install --no-cache-dir networkx==1.11
                                                               
