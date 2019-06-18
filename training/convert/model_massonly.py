from keras.models import load_model

save_path="/local/scratch/ssd2/hborsch/di_tau_mass/DNN/tempsave/"
def main():
    def dummy(y_true, y_pred):
        return y_pred[:,0]
    model = load_model(save_path+"model.h5",
            custom_objects={
                "loss_mass": dummy,
                            })

    with open(save_path+"architecture.json", "w") as f:
        f.write(model.to_json())

    model.save_weights(save_path+"weights.h5")


if __name__ == "__main__":
    main()
