from collab_filtering import Model
import pandas as pd

def create_model() -> Model:
    exp_data = [
        ["a", "b", "c"],
        ["d", "e"],
        ["d", "e", "f"]
    ]
    
    return Model(exp_data)

def test_model_init():
    model = create_model()

    assert isinstance(model, Model)

def test_model_list_recommendation():
    new_data = ["a", "b"]
    model = create_model()

    assert isinstance(model.recommend_items(new_data), pd.Series)


def test_model_recommendation():
    new_data = ["a", "b"]
    model = create_model()

    assert isinstance(model.recommend(new_data), str)