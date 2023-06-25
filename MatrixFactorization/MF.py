
import os
import pprint
import tempfile

from typing import Any, Dict, Text

import numpy as np
import tensorflow as tf

import tensorflow_recommenders as tfrs
#TODO ROZPRACOVÁNO
class MF:
  def __call__(self,ratings, movies):
    # Ratings data.
    ratings = tf.data.Dataset.from_tensor_slices(dict(ratings))
    # %%
    ratings = ratings.map(lambda x: {
        "movie_id": x["itemid"],
        "user_id": x["userid"],
    })
    movies = tf.data.Dataset.from_tensor_slices(dict(movies))
    movies = movies.map(lambda x: x["item_id"])
    tf.random.set_seed(42)
    shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])
    movie_ids = ratings.batch(1_000_000).map(lambda x: x["movie_id"])

    unique_movie_ids = np.unique(np.concatenate(list(movie_ids)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))
    embedding_dimension = 32

    user_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
        # We add an additional embedding to account for unknown tokens.
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    movie_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_movie_ids, mask_token=None),
        tf.keras.layers.Embedding(len(movies) + 1, embedding_dimension)
    ])

    metrics = tfrs.metrics.FactorizedTopK(
        candidates=movies.batch(128).map(movie_model)
    )

    task = tfrs.tasks.Retrieval(
        metrics=metrics,
        #batch_metrics=[tfr.keras.metrics.NDCGMetric()]
    )


    model = MovielensModel(user_model, movie_model)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    cached_train = shuffled.shuffle(100_000).batch(8192).cache()


    model.fit(cached_train, epochs=3)

    scann_index = tfrs.layers.factorized_top_k.ScaNN(model.user_model)
    scann_index.index_from_dataset(
      tf.data.Dataset.zip((unique_movie_ids.batch(100), unique_movie_ids.batch(100).map(model.movie_model)))
    )

    _, titles = scann_index(tf.constant(["42"]))
    print(f"Recommendations for user 42: {titles[0, :3]}")

    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "model")

      # Save the index.
      tf.saved_model.save(
          scann_index,
          path,
          options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
      )

      # Load it back; can also be done in TensorFlow Serving.
      loaded = tf.saved_model.load(path)

      # Pass a user id in, get top predicted movie titles back.
      scores, titles = loaded(["42"])

      print(f"Recommendations: {titles[0][:3]}")
      self.model=model

class MovielensModel(tfrs.Model):

  def __init__(self, user_model, movie_model, task):
    super().__init__()
    self.movie_model: tf.keras.Model = movie_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["user_id"])
    # And pick out the movie features and pass them into the movie model,
    # getting embeddings back.
    positive_movie_embeddings = self.movie_model(features["movie_title"])

    # The task computes the loss and the metrics.
    return self.task(user_embeddings, positive_movie_embeddings)
"""
class NoBaseClassMovielensModel(tf.keras.Model):

  def __init__(self, user_model, movie_model):
    super().__init__()
    self.movie_model: tf.keras.Model = movie_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def train_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

    # Set up a gradient tape to record gradients.
    with tf.GradientTape() as tape:

      # Loss computation.
      user_embeddings = self.user_model(features["user_id"])
      positive_movie_embeddings = self.movie_model(features["movie_title"])
      loss = self.task(user_embeddings, positive_movie_embeddings)

      # Handle regularization losses as well.
      regularization_loss = sum(self.losses)

      total_loss = loss + regularization_loss

    gradients = tape.gradient(total_loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics["loss"] = loss
    metrics["regularization_loss"] = regularization_loss
    metrics["total_loss"] = total_loss

    return metrics

  def test_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

    # Loss computation.
    user_embeddings = self.user_model(features["user_id"])
    positive_movie_embeddings = self.movie_model(features["movie_title"])
    loss = self.task(user_embeddings, positive_movie_embeddings)

    # Handle regularization losses as well.
    regularization_loss = sum(self.losses)

    total_loss = loss + regularization_loss

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics["loss"] = loss
    metrics["regularization_loss"] = regularization_loss
    metrics["total_loss"] = total_loss

    return metrics
"""