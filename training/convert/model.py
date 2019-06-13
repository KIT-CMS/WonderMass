from keras.models import load_model


def main():
    def dummy(y_true, y_pred):
        return y_pred[:,0]
    model = load_model("model.h5",
            custom_objects={
                "loss_p3_1": dummy,
                "loss_p3_2": dummy,
                "loss_p3_h": dummy,
                "loss_f_1": dummy,
                "loss_f_2": dummy,
                "loss_p3_h": dummy,
                "loss_p_1": dummy,
                "loss_p_2": dummy,
                "loss_p_h": dummy,
                "loss_mass": dummy,
                "loss_pt": dummy,
                "loss_eta": dummy,
                "loss_phi": dummy,
                "loss_p": dummy,
                "loss": dummy,
                            })

    with open("architecture.json", "w") as f:
        f.write(model.to_json())

    model.save_weights("weights.h5")


if __name__ == "__main__":
    main()
