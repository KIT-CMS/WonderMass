from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def make_prediction(x_data,export_dir="/local/scratch/ssd2/hborsch/di_tau_mass/DNN/Wondermass/saved_models/best_models/final_lbn_2_layers_1_8"):
    #x_data is np.array of Taus and MET (and jets) in the following shape: (number_events,number_particles=3(+4),dimension_fourvectors=4)
    #where the particle number corresponds to the first tau,second tau, MET (,jets)
    #and the fourvectors are the four components of them: (E,px,py,pz)
    #note that in case of the MET pz should be chosen to pz=0 and E has to be calculated under the assumption that m=0
    #this means E=sqrt(px**2+py**2)

    #returns: np.array of predicted masses of shape (n_events)

    x_test=x_data
    #generate some labels that are needed for the initializer but are not used for the prediction
    y_test=np.zeros(x_test.shape[0])
    #ConfigProto(gpu_options=gpu_options) needed for running tensorflow on gpu
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = tf.saved_model.loader.load(export_dir=export_dir, sess=sess, tags=[tag_constants.SERVING])
        graph = tf.get_default_graph()
        #print(graph.get_operations())
        x_t = graph.get_tensor_by_name("x_t:0")
        y_t=graph.get_tensor_by_name("y_t:0")
        is_training_t=graph.get_tensor_by_name("my_bool:0")
        training_init=graph.get_operation_by_name("dataset_init")
        output=graph.get_tensor_by_name("output/BiasAdd:0")

        mean_t=np.load(open(export_dir+"/mean", "rb"))
        std_t=np.load(open(export_dir+"/std", "rb"))

        sess.run(training_init, feed_dict={
            x_t: x_test,  # feeling the placeholders
            y_t: y_test,  # feeling the placeholders
        })
        pred_val_list=[]
        while True:
            try:
                pred_val = sess.run(output, feed_dict={is_training_t: False})
                pred_val_list.append(pred_val)
            except tf.errors.OutOfRangeError:
                break

        prediction = np.concatenate(pred_val_list)
        prediction=np.squeeze(prediction)

        return prediction

#predict masses (y_values) for the test data
x_test_data=np.load(open("/local/scratch/ssd2/hborsch/di_tau_mass/DNN/Wondermass/saved_models/saves_lbn_2_layer_29_07/x_test","rb"))

#import time
#start_time = time.time()
y_test_pred=make_prediction(x_test_data) #be aware of the fact that y_test_pred is shorter than y_train, because the rest of the batch is dropped
#print("--- %s seconds ---" % (time.time() - start_time))
#print(y_test_pred.shape)
