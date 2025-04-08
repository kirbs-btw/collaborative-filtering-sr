import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

class placeholder: 
    def __init__(self):
        ...


    def train(self, data: list[list[str]]):
        mlb = MultiLabelBinarizer()
        user_item_matrix = pd.DataFrame(mlb.fit_transform(data), columns=mlb.classes_)

        item_similarity = pd.DataFrame(
            cosine_similarity(user_item_matrix.T),
            index=user_item_matrix.columns,
            columns=user_item_matrix.columns
        )


    def recommend(self, target_data):

        # need a way to get the vector of the data in the cf topic 
        # user_vector = user_item_matrix.iloc[user_id]
        # scores = item_similarity.dot(user_vector)
        # scores = scores[~user_vector.astype(bool)]  # Remove already seen items

        # recommended_items = scores.sort_values(ascending=False)
        pass