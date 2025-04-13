import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

class Model: 
    def __init__(self, data: list[list[str]]):
        self.__mlb = MultiLabelBinarizer()
        self.item_similarity = self.train(data)


    def train(self, data: list[list[str]]):
        user_item_matrix = pd.DataFrame(self.__mlb.fit_transform(data), columns=self.__mlb.classes_)

        return pd.DataFrame(
            cosine_similarity(user_item_matrix.T),
            index=user_item_matrix.columns,
            columns=user_item_matrix.columns
        )

    def recommend_items(self, target_data: list[str]):
        new_vector = pd.Series(self.__mlb.transform([target_data])[0], index=self.__mlb.classes_)
        scores = self.item_similarity.dot(new_vector)
        scores = scores[~new_vector.astype(bool)]  # Remove already seen items
        recommended_items = scores.sort_values(ascending=False)
        return recommended_items

    def recommend(self, target_data: list[str]):
        return self.recommend_items(target_data).idxmax()