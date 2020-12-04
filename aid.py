from multiprocessing import Process

import pyttsx3
import tensorflow as tf

# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

import re
import numpy as np
import os
from PIL import Image
import pickle
import numpy as np
import cv2
import time

# define video path -- to be switched to live channel later
from player import play_video, speak

video_path = 'sample.mp4'
# define the interval after which a frame is selected - the nth frame
frame_interval = 160
# captions
old_caption = ""
new_caption = ""
# checkpoint_path = "/gdrive/checkpoints/train"        #use if on same drive account as training

def checkpoint_check():
    global checkpoint_path
    checkpoint_path = "./checkpoints/train"
    if not os.path.exists(checkpoint_path):
        assert (False)


# precprocess with inceptionV3

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


# initialize inceptionV3 load pretrained weights on imagenet

def initialize_inceptionv3():
    global image_features_extract_model
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


# initialize tokenizer

def initialize_tokenizer():
    global top_k, tokenizer
    top_k = 5000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer = pickle.load(open("./pickles/tokeniser.pkl", "rb"))


# hyper parameters

def initialize_hyperparameters():
    global top_k, embedding_dim, units, vocab_size, attention_features_shape, max_length
    top_k = 5000
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    embedding_dim = 256
    units = 512
    vocab_size = top_k + 1
    # num_steps = len(img_name_train) // BATCH_SIZE
    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    # These two variables represent that vector shape
    features_shape = 2048
    attention_features_shape = 64
    max_length = 49


# Load the numpy files
def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

        # self.bi = tf.keras.layers.LSTM(self.units,
        #                               return_sequences=True,
        #                               return_state=True,
        #                               recurrent_initializer='glorot_uniform')
        # self.fc0 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.units, activation='sigmoid'))
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # x = self.fc0(output)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


def define_model_components():
    global encoder, decoder, optimizer, loss_object
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')





def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def load_checkpoint():
    global ckpt, ckpt_manager
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)
    else:
        assert (False)


def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        # attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result

        dec_input = tf.expand_dims([predicted_id], 0)

    # attention_plot = attention_plot[:len(result), :]
    return result


def test(image_path):
    # image_url = url
    #ckpt.restore(ckpt_manager.latest_checkpoint)
    # image_extension = image_url[-4:]
    # image_path = tf.keras.utils.get_file('image' + image_extension, origin=image_url)
    result = evaluate(image_path)
    # print(str(image_path) + "in test")
    # print('Prediction Caption:', ' '.join(result))
    # plot_attention(image_path, result, attention_plot)
    # opening the image
    Image.open(image_path)
    return ' '.join(result[:-1])


def initialize():
    global eng
    checkpoint_check()
    initialize_inceptionv3()
    initialize_tokenizer()
    initialize_hyperparameters()
    define_model_components()
    load_checkpoint()
    eng = pyttsx3.init()
    #print(str(path)+"in generate")


def process_caption(old_caption, new_caption):
    # global checkpoint_path,image_features_extract_model,top_k,tokenizer,embedding_dim, units, vocab_size,attention_features_shape, max_length,encoder, decoder, optimizer, loss_object,ckpt, ckpt_manager
    # global old_caption, new_caption, eng
    # with open('global_state.pickle', 'rb') as handle:
       # dict = pickle.load(handle)
    # globals().update(dict)
    # new_caption = test("frame.jpg")
    print(new_caption)
    eng = pyttsx3.init()
    eng.say(new_caption)

    # Runs for small duration of time otherwise we may not be able to hear
    eng.runAndWait()
    # to do text similarity and text-to-speech part
    #return new_caption


def caption_video(video_path):
    global old_caption, new_caption
    cap = cv2.VideoCapture(video_path)
    #print("running")
    next_frame = 0
    processes = []
    # dict = {"checkpoint_path": checkpoint_path,"image_features_extract_model": image_features_extract_model,
            # "top_k": top_k, "tokenizer": tokenizer}
    # with open('global_state.pickle', 'wb') as handle:
        # pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    while (cap.isOpened()):
        #print("running loop")

        next_frame = next_frame + 1
        print(next_frame)
        #if old_caption is "" and next_frame == 1:
        #    print("waiting")
        #    time.sleep(6.7)
        #    continue

        if old_caption is "" and next_frame < frame_interval:
            ret = cap.grab()
            time.sleep(41 / 1000)
            continue
        else:
            if next_frame != frame_interval:
                ret = cap.grab()
                time.sleep(41 / 1000)
                continue
            if next_frame == frame_interval:
                ret, frame = cap.read()
                # print("processing frame")
                next_frame = 0
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (50, 50)
                fontScale = 1
                fontColor = (255, 255, 255)
                lineType = 2
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite("frame.jpg", frame)
                start = time.process_time()
                new_caption = test("frame.jpg")
                if old_caption is "":
                    old_caption = new_caption

                # p = Process(target=process_caption, args=(old_caption, new_caption))
                # p.start()
                # print(p.pid)
                # processes.append(p)
                # new_caption="jjh"
                # print(new_caption)
                cv2.putText(frame, new_caption, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
                imS = cv2.resize(frame, (530, 300))

                process_caption(old_caption, new_caption)
                print(time.process_time() - start)
                cv2.imshow('frame', imS)

        #cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    # for pr in processes:
        # pr.join()





if __name__ == '__main__':
    initialize()
    #p = Process(target=play_video)
    #p.start()
    caption_video(video_path)
    #p.join()

