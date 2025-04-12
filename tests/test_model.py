from collab_filtering import Model

def test_model_init():
    exp_data = [
    ["a", "b", "c"],
    ["d", "e"],
    ["d", "e", "f"]
    ]

    new_data = ["a", "b"]

    model = Model(exp_data)
    assert isinstance(model, Model)

    model.recommend_items(new_data)
    # asst