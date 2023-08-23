import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def truncate_unused_columns(df_post, df_post_rating, df_post_category, df_rating):
    df_post_truncated = df_post[["id", "post_category_id", "title"]]
    df_post_rating_truncated = df_post_rating[["id", "post_id"]]
    df_post_category_truncated = df_post_category[["id", "name", "description", "slug"]]
    df_rating_truncated = df_rating[["id", "post_rating_id", "user_id", "rating_score"]]

    return df_post_truncated, df_post_rating_truncated, df_post_category_truncated, df_rating_truncated


def combine_post_and_post_rating(df_post, df_post_rating):
    # Combine `posts` and `post_ratings`
    df_post_combined = pd.merge(df_post, df_post_rating, left_on="id", right_on="post_id")

    # Remove id_x (posts#id) & (post_ratings#post_id)
    df_post_combined = df_post_combined[["id_y", "post_category_id", "title"]]

    # Rename id_x to id
    df_post_combined = df_post_combined.rename(columns={"id_y": "id"})

    return df_post_combined


def clean_post_category(df_post_category, df_post):
    df_post_category_cleaned = df_post_category.dropna()

    # Delete post_category 22 since there is a duplicate
    df_post_category_cleaned = df_post_category_cleaned[df_post_category_cleaned["id"] != 22]

    # Remove post_category that has 0 post
    post_category_ids = df_post["post_category_id"].unique()
    df_post_category_cleaned = df_post_category_cleaned[df_post_category_cleaned["id"].isin(post_category_ids)]

    return df_post_category_cleaned


def replace_post_category_in_post(df_post):
    df_post_replaced = df_post.copy()
    df_post_replaced.loc[df_post_replaced["post_category_id"] == 22, "post_category_id"] = 21

    return df_post_replaced


def rename_rating_table(df_rating):
    df_rating_renamed = df_rating.copy()
    df_rating_renamed = df_rating_renamed.rename(columns={"post_rating_id": "post_id"})

    return df_rating_renamed


def get_authentic_user(df_rating):
    df_rating_authentic = df_rating.copy()
    df_rating_authentic = df_rating_authentic.dropna(subset=["user_id"])

    # Create user data frames
    df_user = df_rating_authentic[["user_id"]].copy()
    df_user.rename(columns={"user_id": "id"})

    return df_rating_authentic, df_user


def create_df_user_train_set(df_post, df_post_category, df_rating):
    # Merge rating with post & post categories
    df_user_train_set = pd.merge(df_rating, df_post, left_on="post_id", right_on="id")
    df_user_train_set = pd.merge(df_user_train_set, df_post_category, left_on="post_category_id", right_on="id")
    df_user_train_set = df_user_train_set[["user_id", "rating_score", "slug"]]
    df_user_train_set = df_user_train_set.rename(columns={"user_id": "id"})

    # Get unique slugs
    unique_slugs = df_post_category["slug"].unique()

    # Group by user_id and slug, to calculate the mean rating_score
    df_user_train_set = df_user_train_set.groupby(["id", "slug"]).mean().unstack(level=1)
    df_user_train_set.columns = df_user_train_set.columns.to_flat_index().map(lambda x: x[1])

    # Replace NaN with 0.0
    df_user_train_set = df_user_train_set.fillna(0)

    # Build user vector X_u by merging ratings_df & user_features
    df_user_train_set = df_rating.merge(df_user_train_set, left_on="user_id", right_on="id")

    # Arrange columns to user_id in front
    user_id_column = df_user_train_set.pop("user_id")
    df_user_train_set.insert(0, "user_id", user_id_column)

    # Only retain columns in unique_slugs
    columns = np.append(["user_id"], unique_slugs)
    df_user_train_set = df_user_train_set[columns]

    return df_user_train_set


def create_df_item_train_set(df_post, df_post_category, df_rating):
    df_item_train_set = pd.merge(df_post, df_post_category, left_on="post_category_id", right_on="id")
    df_item_train_set = df_item_train_set[["id_x", "slug"]]
    df_item_train_set = df_item_train_set.rename(columns={"id_x": "id"})

    # Build one-hot-encoding with slug
    df_item_train_set = pd.get_dummies(df_item_train_set, columns=["slug"], prefix="", prefix_sep="")

    # Build article vector X_a by merging ratings_df & article_features
    df_item_train_set = df_rating.merge(df_item_train_set, left_on="post_id", right_on="id")

    # Select only slugs & post id
    unique_slugs = df_post_category["slug"].unique()
    columns = np.append(unique_slugs, ["post_id"])
    df_item_train_set = df_item_train_set[columns]

    # Calculate average rating & total ratings
    df_item_train_set_temp = df_rating[["post_id", "rating_score"]]
    df_item_train_set_temp = df_item_train_set_temp.groupby("post_id").agg({"rating_score": ["mean", "count"]})
    df_item_train_set_temp.columns = df_item_train_set_temp.columns.to_flat_index().map(lambda x: x[1])

    # Merge both to get average and total rating on main dataframe
    df_item_train_set = pd.merge(df_item_train_set, df_item_train_set_temp, left_on="post_id", right_on="post_id")

    # Rename average and total ratings
    df_item_train_set = df_item_train_set.rename(columns={"mean": "average", "count": "total_ratings"})

    # Arrange columns to move post_id, average and total ratings in front
    average_columns = df_item_train_set.pop("average")
    df_item_train_set.insert(0, "average", average_columns)

    total_ratings_column = df_item_train_set.pop("total_ratings")
    df_item_train_set.insert(0, "total_ratings", total_ratings_column)

    post_id_column = df_item_train_set.pop("post_id")
    df_item_train_set.insert(0, "post_id", post_id_column)

    return df_item_train_set


def create_df_label_train_set(df_rating):
    df_label_train_set = df_rating[["rating_score"]]

    return df_label_train_set


def get_numpy_train_set(dataframe, column_start):
    dataframe_copy = dataframe.copy()
    dataframe_copy = dataframe_copy.iloc[:, column_start:]

    return dataframe_copy.to_numpy()


def build_model(user_train_set, item_train_set, vector_size=16, seed=1):
    tf.random.set_seed(seed)

    # Build user neural network
    user_neural_network = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(vector_size)
    ])
    user_input = tf.keras.layers.Input(shape=(user_train_set.shape[1]))
    user_representation = user_neural_network(user_input)
    user_representation = tf.linalg.l2_normalize(user_representation, axis=1)

    # Build item neural network
    item_neural_network = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(vector_size),
    ])
    item_input = tf.keras.layers.Input(shape=(item_train_set.shape[1]))
    item_representation = item_neural_network(item_input)
    item_representation = tf.linalg.l2_normalize(item_representation, axis=1)

    # Compute the dot product of user vector and item vector
    output = tf.keras.layers.Dot(axes=1)([user_representation, item_representation])

    # Specify the inputs and output of the model
    model = tf.keras.Model([user_input, item_input], output)

    return model


def get_user_vector(user_id, df_post, df_post_category, df_rating):
    # Merge rating with post & post categories
    df_user_train_set = pd.merge(df_rating, df_post, left_on="post_id", right_on="id")
    df_user_train_set = pd.merge(df_user_train_set, df_post_category, left_on="post_category_id", right_on="id")
    df_user_train_set = df_user_train_set[["user_id", "rating_score", "slug"]]
    df_user_train_set = df_user_train_set.rename(columns={"user_id": "id"})

    # Get unique slugs
    unique_slugs = df_post_category["slug"].unique()

    # Group by user_id and slug, to calculate the mean rating_score
    df_user_train_set = df_user_train_set.groupby(["id", "slug"]).mean().unstack(level=1)
    df_user_train_set.columns = df_user_train_set.columns.to_flat_index().map(lambda x: x[1])

    # Replace NaN with 0.0
    df_user_train_set = df_user_train_set.fillna(0)

    # Build user vector X_u by merging ratings_df & user_features
    df_user_train_set = df_rating.merge(df_user_train_set, left_on="user_id", right_on="id")

    # Arrange columns to user_id in front
    user_id_column = df_user_train_set.pop("user_id")
    df_user_train_set.insert(0, "user_id", user_id_column)

    # Only retain columns in unique_slugs
    columns = np.append(["user_id"], unique_slugs)
    df_user_train_set = df_user_train_set[columns]

    # Filter to only get specified user
    df_user_train_set = df_user_train_set[df_user_train_set["user_id"] == user_id]

    if df_user_train_set.empty:
        return None

    return df_user_train_set.to_numpy()[0]


def generate_user_vectors(user_vector, vector_size):
    user_vectors = np.repeat([user_vector], [vector_size], axis=0)

    return user_vectors


def generate_item_vectors(df_post, df_post_category, df_rating):
    df_item_train_set = pd.merge(df_post, df_post_category, left_on="post_category_id", right_on="id")
    df_item_train_set = df_item_train_set[["id_x", "slug"]]
    df_item_train_set = df_item_train_set.rename(columns={"id_x": "id"})

    # Build one-hot-encoding with slug
    df_item_train_set = pd.get_dummies(df_item_train_set, columns=["slug"], prefix="", prefix_sep="")

    # Build article vector X_a by merging ratings_df & article_features
    df_item_train_set = df_rating.merge(df_item_train_set, left_on="post_id", right_on="id")

    # Select only slugs & post id
    unique_slugs = df_post_category["slug"].unique()
    columns = np.append(unique_slugs, ["post_id"])
    df_item_train_set = df_item_train_set[columns]

    # Calculate average rating & total ratings
    df_item_train_set_temp = df_rating[["post_id", "rating_score"]]
    df_item_train_set_temp = df_item_train_set_temp.groupby("post_id").agg({"rating_score": ["mean", "count"]})
    df_item_train_set_temp.columns = df_item_train_set_temp.columns.to_flat_index().map(lambda x: x[1])

    # Merge both to get average and total rating on main dataframe
    df_item_train_set = pd.merge(df_item_train_set, df_item_train_set_temp, left_on="post_id", right_on="post_id")

    # Get only unique post
    df_item_train_set = df_item_train_set.drop_duplicates(subset=["post_id"])

    # Rename average and total ratings
    df_item_train_set = df_item_train_set.rename(columns={"mean": "average", "count": "total_ratings"})

    # Arrange columns to move post_id, average and total ratings in front
    average_columns = df_item_train_set.pop("average")
    df_item_train_set.insert(0, "average", average_columns)

    total_ratings_column = df_item_train_set.pop("total_ratings")
    df_item_train_set.insert(0, "total_ratings", total_ratings_column)

    post_id_column = df_item_train_set.pop("post_id")
    df_item_train_set.insert(0, "post_id", post_id_column)

    return df_item_train_set.to_numpy()


def get_predicted_item_details(item_vectors, label_vectors, df_post, df_post_category, df_post_rating_raw):
    # Get post_ids and predicted_ratings
    post_ids = item_vectors[:, 0]
    predicted_ratings = label_vectors.flatten()
    data = {"post_id": post_ids, "predicted_rating": predicted_ratings}

    # Create dataframe
    df_predicted_items = pd.DataFrame(data)

    # Merge with post and post_category and post_ratings to get details
    df_predicted_items = pd.merge(df_predicted_items, df_post, left_on="post_id", right_on="id")
    df_predicted_items = pd.merge(df_predicted_items, df_post_category, left_on="post_category_id", right_on="id")
    df_predicted_items = pd.merge(df_predicted_items, df_post_rating_raw, left_on="post_id", right_on="id")

    df_predicted_items = df_predicted_items.rename(columns={"post_id_y": "post_id"})
    df_predicted_items = df_predicted_items.rename(columns={"authenticated_average": "average_rating"})

    df_predicted_items = df_predicted_items[["post_id", "predicted_rating", "average_rating", "title", "slug"]]

    return df_predicted_items


def predict_rating_with_vectors(user_vectors, item_vectors):
    # Scale both user_vectors and item_vectors
    scaled_user_vectors = user_scaler.transform(user_vectors[:, user_column_start:])
    scaled_item_vectors = item_scaler.transform(item_vectors[:, item_column_start:])

    # Make prediction
    predicted_label_vectors = model.predict([scaled_user_vectors, scaled_item_vectors])

    # Unscale prediction
    predicted_label_vectors = label_scaler.inverse_transform(predicted_label_vectors)

    # Sorting label vectors and item vectors
    sorted_index = np.argsort(-predicted_label_vectors, axis=0).reshape(-1).tolist()
    predicted_label_vectors = predicted_label_vectors[sorted_index]
    item_vectors = item_vectors[sorted_index]

    # Get item details
    df_predicted_items = get_predicted_item_details(item_vectors, predicted_label_vectors, df_post, df_post_category,
                                                    df_post_rating_raw)
    df_predicted_items = df_predicted_items.sort_values(by="predicted_rating", ascending=False)

    return df_predicted_items


def get_recommended_posts_for(user_id):
    # Get item_vectors
    item_vectors = generate_item_vectors(df_post, df_post_category, df_rating)

    # Get user_vectors
    user_vector = get_user_vector(user_id, df_post, df_post_category, df_rating)

    # Error handling for null user
    if user_vector is None:
        return []

    user_vectors = generate_user_vectors(user_vector, item_vectors.shape[0])

    # Get predicted ratings
    df_predicted_items = predict_rating_with_vectors(user_vectors, item_vectors)
    recommended_post_ids = df_predicted_items["post_id"].tolist()
    predicted_ratings = df_predicted_items["predicted_rating"].tolist()

    return [recommended_post_ids, predicted_ratings]


# Read data from raw CSV files
df_post_raw = pd.read_csv("data/posts.csv")
df_post_rating_raw = pd.read_csv("data/post_ratings.csv")
df_post_category_raw = pd.read_csv("data/post_categories.csv")
df_rating_raw = pd.read_csv("data/ratings.csv")

# Get truncated dataframes
df_post, df_post_rating, df_post_category, df_rating = truncate_unused_columns(df_post_raw, df_post_rating_raw,
                                                                               df_post_category_raw, df_rating_raw)

# Combine `posts` with `post_ratings` since it is a one-to-one relationship
df_post = combine_post_and_post_rating(df_post, df_post_rating)

# Clean up duplicate in `post_categories` & replace duplicate in `posts`
df_post_category = clean_post_category(df_post_category, df_post)
df_post = replace_post_category_in_post(df_post)

# Rename post_rating_id to post_id in `ratings`
df_rating = rename_rating_table(df_rating)

# Get authentic users
df_rating, df_user = get_authentic_user(df_rating)

# Create user, item and label train set
df_user_train_set = create_df_user_train_set(df_post, df_post_category, df_rating)
df_item_train_set = create_df_item_train_set(df_post, df_post_category, df_rating)
df_label_train_set = create_df_label_train_set(df_rating)

# Columns to start with
user_column_start = 1
item_column_start = 1
label_column_start = 0

# Convert dataframe to numpy
user_train_set = get_numpy_train_set(df_user_train_set, user_column_start)
item_train_set = get_numpy_train_set(df_item_train_set, item_column_start)
label_train_set = get_numpy_train_set(df_label_train_set, label_column_start)

# Scale User
user_scaler = StandardScaler()
user_scaler.fit(user_train_set)
user_train_set = user_scaler.transform(user_train_set)

# Scale Item
item_scaler = StandardScaler()
item_scaler.fit(item_train_set)
item_train_set = item_scaler.transform(item_train_set)

# Scale Label
label_scaler = MinMaxScaler((-1, 1))
label_scaler.fit(label_train_set.reshape(-1, 1))
label_train_set = label_scaler.transform(label_train_set)

# Split train set
user_train_set, user_test_set = train_test_split(user_train_set, train_size=0.80, shuffle=True, random_state=1)
item_train_set, item_test_set = train_test_split(item_train_set, train_size=0.80, shuffle=True, random_state=1)
label_train_set, label_test_set = train_test_split(label_train_set, train_size=0.80, shuffle=True, random_state=1)
model = build_model(user_train_set, item_train_set)
model.summary()

# Configuration
tf.random.set_seed(1)
cost_function = tf.keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.1)

# Compile Model
model.compile(optimizer=optimizer, loss=cost_function)
model.fit([user_train_set, item_train_set], label_train_set, epochs=30)
