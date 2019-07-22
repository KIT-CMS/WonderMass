from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import tensorflow as tf
from sklearn.model_selection import train_test_split
import uproot
from lbn import LBN
from shutil import rmtree

#path of root file for training, tree name
path = "MSSM_4jets.root"
tree_name="Events_tt"

# constants
BATCH_SIZE = 10000
LEARNING_RATE = 1e-3
n_epochs=3000


# def resample(y_data,n=321):
#     #bins mass indices in n bins and copies randomly mass indices from every bin until it has the same number of events as the highest bin.
#     #small binning makes resampling better, but it will fail when bin does not contain at least one event.
#     #takes much computing time, should be made faster, probably is not even beneficial for training
#     p = np.percentile(y_data, [0.5, 99.5])
#     bins=np.linspace(p[0],p[1],num=n)
#     """
#     plt.figure()
#     plt.hist(y_data,bins=5)
#     plt.savefig("testhistbefore")
#     """
#     indices=[]
#     for i in range(len(bins)-1):
#         indexlist=[]
#         for n,mass in enumerate(y_data):
#             if bins[i]<=mass and mass<bins[i+1]:
#                 indexlist.append(n)
#         indices.append(indexlist)
#
#     count=0
#     for indexlist in indices:
#         if len(indexlist)>count:
#             count=len(indexlist)
#     print("maximum samplenumber per bin=",count)
#     indices_res=[]
#     for indexlist in indices:
#         indexlist+=list(np.random.choice(indexlist,count-len(indexlist)))
#         indices_res+=indexlist
#
#     print("Min/max resampling mass range: {}/{}".format(p[0], p[-1]))
#     from sklearn.model_selection import train_test_split
#     idx_re = np.array(indices_res)
#     """
#     plt.figure()
#     plt.hist(y_data[idx_train_re],bins=4)
#     plt.savefig("testhistafter.png")
#     """
#     return idx_re


#reading in root file
components = ["e","px", "py", "pz"]
inputs_t1=["t1_rec_" + c for c in components]
inputs_t2=["t2_rec_" + c for c in components]
inputs_met=["met_rec_" + c for c in ["px", "py"]]
inputs_jet1=["v_jet1_rec_" + c for c in components]
inputs_jet2=["v_jet2_rec_" + c for c in components]
inputs_jet3=["v_jet3_rec_" + c for c in components]
inputs_jet4=["v_jet4_rec_" + c for c in components]

data = {}
tree = uproot.open(path)["ntupleBuilder"][tree_name].arrays()
#tree = uproot.open(path)[tree_name].arrays()
for b in tree:
    data[b.decode("utf-8")] = tree[b]

#x_data
#def vecs of taus and MET
vec_t1=np.stack([data[name] for name in inputs_t1],axis=-1)
vec_t2=np.stack([data[name] for name in inputs_t2],axis=-1)
vec_jet1=np.stack([data[name] for name in inputs_jet1],axis=-1)
vec_jet2=np.stack([data[name] for name in inputs_jet2],axis=-1)
vec_jet3=np.stack([data[name] for name in inputs_jet3],axis=-1)
vec_jet4=np.stack([data[name] for name in inputs_jet4],axis=-1)

#define Energy of vec_met as E=sqrt(px**2+py**2) and pz=0
E_met=np.sqrt(data["met_rec_px"]**2+data["met_rec_py"]**2)
pz_met=np.zeros(len(E_met))
list_met=[E_met,data["met_rec_px"],data["met_rec_py"],pz_met]
vec_met=np.stack(list_met,axis=-1)

#shape of x_data: nEvents, nParticles=3, 4
x_data=np.stack([vec_t1,vec_t2,vec_met,vec_jet1,vec_jet2,vec_jet3,vec_jet4],axis=1)
#y_data
y_data=data["h_gen_mass"]

#resampling events to flat mass, makes training much slower and should not be very benificial
#idx=resample(y_data,321)
#x_data=x_data[idx,:,:]
#y_data=y_data[idx]

#normalize outputs
mean_t = np.mean(y_data, axis=0)
std_t = np.std(y_data, axis=0)

# TENSORFLOW ####################################
# split train and test data
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2,random_state=1234)

# create a tensorflow session
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# Define placeholders
is_training_t = tf.placeholder(tf.bool,name="my_bool")

x_t = tf.placeholder(
    tf.float32,
    shape=[None, 7, 4],
    name="x_t"
)
y_t = tf.placeholder(
    tf.float32,
    shape=[None],
    name="y_t"
)

# create `tf.data.Dataset`, build batches and make it iterable
dataset = tf.data.Dataset.from_tensor_slices((x_t, y_t))
dataset = dataset.batch(BATCH_SIZE,drop_remainder=True)
iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
training_init = iterator.make_initializer(dataset, name='dataset_init')
next_batch= iterator.get_next()

# initialize the LBN, set 6 combinations
lbn = LBN(6, boost_mode=LBN.PAIRS, batch_norm=True, is_training=is_training_t)


# # create a feature tensor based on input four-vectors
# # and register it to the feature factory, here:
# # m**2 = E**2 - p**2 of all particle combinations
# @lbn.register_feature
# def pair_m2(factory):
#     all_pair_m2 = factory.E**2 - factory.p**2
#     return tf.gather(tf.reshape(all_pair_m2, [-1, factory.n**2]), factory.triu_indices, axis=1)


# extract features out of the lbn
features = lbn(next_batch[0])  # Performs all the Lorenz Boost Network

# Build DNN model for mass regression
hidden = tf.layers.dense(
    #tf.reshape(next_batch[0],[BATCH_SIZE,-1]),      #only for training without lbn (instead of features)
    features,
    500,
    activation=tf.nn.elu,
)
hidden=tf.layers.dropout(
hidden,
rate=0.3,
training=is_training_t,
)
for i in range(3):
    hidden = tf.layers.dense(
        hidden,
        500,
        activation=tf.nn.elu,
    )
    hidden = tf.layers.batch_normalization(hidden, training=is_training_t)
    hidden=tf.layers.dropout(
    hidden,
    rate=0.3,
    training=is_training_t,
    )

output = tf.layers.dense(hidden, 1, activation=None,name="output")  # 1 output node: for mass of di-tau system

num_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
print("Model has {} trainable parameters!".format(num_params))

updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

loss = tf.losses.mean_squared_error(tf.squeeze(next_batch[1]), tf.squeeze(output),weights=1/tf.squeeze(next_batch[1])**2)

trainer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss=loss)

# set the weights and biases of the NN to random initial values which follow the kernel_initializer
init = tf.global_variables_initializer()  # (kernel==weights)
sess.run(init)

sess.run(training_init, feed_dict={
    x_t: x_train,  # feeling the placeholders
    y_t: y_train,  # feeling the placeholders
})

# run the training with `is_training_t: True`
train_loss_results = []
test_loss_results = []


# Trainining with n_epochs
min_loss=99999
for i in tqdm(range(n_epochs)):
    print("\n\033[95m[EPOCH]\033[0;0m: {}".format(i))
    while True:  # feed to one epoch all the data in batches...
        try:
            train, loss_train, _ = sess.run([trainer, loss, updates], feed_dict={is_training_t: True})
            loss_test = sess.run(loss, feed_dict={is_training_t: False})
        except tf.errors.OutOfRangeError:  # until no more batches left to feed
            sess.run(training_init, feed_dict={
                x_t: x_train,  # feeling the placeholders
                y_t: y_train,  # feeling the placeholders
            })
            break
    print("\033[1;32m[TRAIN]\033[0;0m loss: {}".format(loss_train))
    print("\033[1;33m[TEST]\033[0;0m loss: {} \n".format(loss_test))
    train_loss_results.append(loss_train)
    test_loss_results.append(loss_test)

    #save best training
    if len(test_loss_results)!=1:
        if test_loss_results[i]<min(test_loss_results[0:i]):
            rmtree('savemodeldir', ignore_errors=True)
            tf.saved_model.simple_save(sess, export_dir="savemodeldir",
                                            inputs={"x_t": x_t, "y_t": y_t,
                                                    "isTrainingBool": is_training_t},
                                                    outputs={"output": output})
            particle_weights=sess.run(lbn.particle_weights)
            restframe_weights=sess.run(lbn.restframe_weights)
            np.save(open("savemodeldir/particle_weights","wb"), particle_weights)
            np.save(open("savemodeldir/restframe_weights","wb"), restframe_weights)
            min_loss=test_loss_results[i]
            print("test loss has improved")
        else:
            print("test loss has not improved from",min_loss)
    pass

#Save the mean and std and test data for plotting results
np.save(open("savemodeldir/mean","wb"),mean_t)
np.save(open("savemodeldir/std","wb"),std_t)
np.save(open("savemodeldir/x_test","wb"), x_test)
np.save(open("savemodeldir/y_test","wb"), y_test)

np.save(open("savemodeldir/train_loss", "wb"), train_loss_results)
np.save(open("savemodeldir/test_loss", "wb"), test_loss_results)

#Evaluating the model on test dataset, saving the result

def make_prediction(x_data,export_dir="savemodeldir"):
    #x_data is np.array of Taus and MET in the following shape: (number_events,number_particles=3,dimension_fourvectors=4)
    #where the particle number corresponds to the first tau,second tau, MET
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
        #prediction = prediction * std_t + mean_t
        prediction=np.squeeze(prediction)

        return prediction

#predict masses (y_values) for the test data
x_test_data=np.load(open("savemodeldir/x_test","rb"))
y_test_pred=make_prediction(x_test_data,export_dir="savemodeldir") #be aware of the fact that y_test_pred is shorter than y_train, because the rest of the batch is dropped

np.save(open("savemodeldir/y_test_pred","wb"), y_test_pred)
