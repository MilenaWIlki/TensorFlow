import tensorflow as tf
from tensorflow.keras.layers import Embedding, Flatten, Concatenate, Dense, Input
from tensorflow.keras.models import Model

# Sample user-item interaction data
num_users = 1000
num_items = 500
user_ids = np.random.randint(1, num_users, size=10000)
item_ids = np.random.randint(1, num_items, size=10000)
ratings = np.random.randint(1, 6, size=10000)

# Define NCF model
user_input = Input(shape=(1,))
item_input = Input(shape=(1,))
user_embedding = Embedding(num_users + 1, 32)(user_input)
item_embedding = Embedding(num_items + 1, 32)(item_input)
user_flatten = Flatten()(user_embedding)
item_flatten = Flatten()(item_embedding)
concat = Concatenate()([user_flatten, item_flatten])
dense1 = Dense(64, activation='relu')(concat)
output = Dense(1)(dense1)
model = Model(inputs=[user_input, item_input], outputs=output)
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit([user_ids, item_ids], ratings, epochs=10, batch_size=32, validation_split=0.2)

# Generate recommendations for users
user_id = 123
items = np.arange(1, num_items + 1)
predictions = model.predict([np.array([user_id] * num_items), items])
top_items = items[np.argsort(predictions[:, 0])[::-1][:5]]
print("Recommendations for user", user_id, ":", top_items)
