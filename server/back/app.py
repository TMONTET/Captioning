import os.path
import pickle

from flask import Flask, request, abort, send_file
import tensorflow as tf
from flask_cors import CORS
from keras import Model
from keras_preprocessing.image import load_img, img_to_array, save_img
from skimage.util import random_noise

app = Flask(__name__)
CORS(app)

embedding_dim = 256
units = 512
vocab_size = 5001
max_length = 31

with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

class Encoder(Model):
    def __init__(self, embed_dim):
        super(Encoder, self).__init__()
        self.dense = tf.keras.layers.Dense(embed_dim)  # build your Dense layer with relu activation

    def call(self, features):
        features = self.dense(features)  # extract the features from the image shape: (batch, 8*8, embed_dim)
        features = tf.keras.activations.relu(features, alpha=0.01, max_value=None, threshold=0)
        return features


encoder = Encoder(embedding_dim)

class Attention_model(Model):
    def __init__(self, units):
        super(Attention_model, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)  # build your Dense layer
        self.W2 = tf.keras.layers.Dense(units)  # build your Dense layer
        self.V = tf.keras.layers.Dense(1)  # build your final Dense layer with unit 1
        self.units = units

    def call(self, features, hidden):
        # features shape: (batch_size, 8*8, embedding_dim)
        # hidden shape: (batch_size, hidden_size)

        # Expand the hidden shape to shape: (batch_size, 1, hidden_size)
        hidden_with_time_axis = hidden[:, tf.newaxis]

        # build your score funciton to shape: (batch_size, 8*8, units)
        score = tf.keras.activations.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # extract your attention weights with shape: (batch_size, 8*8, 1)
        attention_weights = tf.keras.activations.softmax(self.V(score), axis=1)

        # shape: create the context vector with shape (batch_size, 8*8,embedding_dim)
        context_vector = attention_weights * features

        # reduce the shape to (batch_size, embedding_dim)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(Model):
    def __init__(self, embed_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units = units
        self.attention = Attention_model(self.units)  # iniitalise your Attention model with units
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim)  # build your Embedding layer
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.d1 = tf.keras.layers.Dense(self.units)  # build your Dense layer
        self.d2 = tf.keras.layers.Dense(vocab_size)  # build your Dense layer

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)  # create your context vector & attention weights from attention model
        embed = self.embed(x)  # embed your input to shape: (batch_size, 1, embedding_dim)
        embed = tf.concat([tf.expand_dims(context_vector, 1), embed],
                          axis=-1)  # Concatenate your input with the context vector from attention layer. Shape: (batch_size, 1, embedding_dim + embedding_dim)
        output, state = self.gru(
            embed)  # Extract the output & hidden state from GRU layer. Output shape : (batch_size, max_length, hidden_size)
        output = self.d1(output)
        output = tf.reshape(output, (-1, output.shape[2]))  # shape : (batch_size * max_length, hidden_size)
        output = self.d2(output)  # shape : (batch_size * max_length, vocab_size)

        return output, state, attention_weights

    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


decoder = Decoder(embedding_dim, units, vocab_size)

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)  #define the optimizer
last_checkpoint = tf.train.latest_checkpoint('checkpoints/flickr8k')
checkpoint1 = tf.train.Checkpoint(encoder=encoder,
                                  decoder=decoder,
                                  optimizer=optimizer)
status1 = checkpoint1.restore(last_checkpoint)
status1.expect_partial()

classifier = tf.keras.models.load_model('models/classification')
autoencoder = tf.keras.models.load_model('models/autoencoder')

image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.compat.v1.keras.Model(new_input, hidden_layer)

@app.post('/verify')
def verify():
    if 'file' not in request.files:
        abort(400, 'No file found')

    file = request.files['file']

    img_path = os.path.join('images', file.filename.replace(' ', '_'))
    file.save(os.path.join(img_path))

    image = load_img(img_path, target_size=(299, 299))
    image = img_to_array(image)
    image = image.reshape(1, 299, 299, 3)
    image = image.astype('float32') / 255.

    predictions = classifier.predict(image)

    prediction = predictions[0][0]
    is_photo = prediction > 0.3
    return {
        'is_photo': 1 if is_photo else 0,
        'probability': 100 * (1 - prediction if is_photo == 0 else prediction)
    }


@app.post('/noise')
def noise():
    if 'file' not in request.files:
        abort(400, 'No file found')

    file = request.files['file']

    img_path = os.path.join('images', file.filename.replace(' ', '_'))
    file.save(os.path.join(img_path))

    original_image = load_img(img_path, target_size=(144, 256))
    original_image = img_to_array(original_image)
    original_image = original_image.reshape(1, 144, 256, 3)
    original_image = original_image.astype('float32') / 255.

    noised_image = random_noise(original_image, mode='gaussian', mean=0, var=0.05)

    noised_path = os.path.join('results', 'noised_' + file.filename.replace(' ', '_'))
    save_img(noised_path, noised_image.reshape(144, 256, 3))
    return send_file(noised_path, attachment_filename='result.jpg')


@app.post('/denoise')
def denoise():
    if 'file' not in request.files:
        abort(400, 'No file found')

    file = request.files['file']

    img_path = os.path.join('images', file.filename.replace(' ', '_'))
    file.save(os.path.join(img_path))

    original_image = load_img(img_path, target_size=(144, 256))
    original_image = img_to_array(original_image)
    original_image = original_image.reshape(1, 144, 256, 3)
    original_image = original_image.astype('float32') / 255.

    noised_image = random_noise(original_image, mode='gaussian', mean=0, var=0.05)

    denoised = autoencoder.predict(noised_image)

    denoised_path = os.path.join('results', 'denoised_' + file.filename.replace(' ', '_'))
    save_img(denoised_path, denoised.reshape(144, 256, 3))
    return send_file(denoised_path, attachment_filename='result.jpg')

@app.post('/caption')
def caption():
    if 'file' not in request.files:
        abort(400, 'No file found')

    file = request.files['file']

    img_path = os.path.join('images', file.filename.replace(' ', '_'))
    file.save(os.path.join(img_path))

    image = tf.io.read_file(img_path, name=None)
    image = tf.image.decode_jpeg(image, channels=0)
    image = tf.image.resize(image, (299, 299))
    image = tf.keras.applications.inception_v3.preprocess_input(image)

    max_length = 31
    hidden = tf.zeros((1, 512))

    temp_input = tf.expand_dims(image, 0)  # process the input image to desired format before extracting features
    img_tensor_val = image_features_extract_model(temp_input)  # Extract features using our feature extraction model
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)  # extract the features by passing the input to encoder

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)  # get the output from decoder

        predicted_id = tf.argmax(
            predictions[0]).numpy()  # extract the predicted id(embedded value) which carries the max value
        # map the id to the word from tokenizer and append the value to the result list
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return {
                "caption": ' '.join(result).rsplit(' ', 1)[0]
            }

        dec_input = tf.expand_dims([predicted_id], 0)

    return {
        "caption": ' '.join(result).rsplit(' ', 1)[0]
    }

if __name__ == '__main__':
    app.run()
