"""
Classify all the images in a holdout set.
"""
import pickle
import sys
import tensorflow as tf
from tqdm import tqdm
import numpy as np


def get_labels():
    """Return a list of our trained labels so we can
    test our training accuracy. The file is in the
    format of one label per line, in the same order
    as the predictions are made. The order can change
    between training runs."""
    with open("./output/retrained_labels2.txt", 'r') as fin:
        labels = [line.rstrip('\n') for line in fin]
    return labels

def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def predict_on_frames(frames, batch):
    """Given a list of frames, predict all their classes."""
    # Unpersists graph from file
    # with tf.gfile.FastGFile("output/retrained_graph.pb", 'rb') as fin:
    #     graph_def = tf.GraphDef()
    #     graph_def.ParseFromString(fin.read())
    #     _ = tf.import_graph_def(graph_def, name='')

    graph = load_graph("output/retrained_graph2.pb")

    with tf.Session(graph=graph) as sess:
        # softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        frame_predictions = []
        image_path = 'frames/'
        pbar = tqdm(total=len(frames))
        for i, frame in enumerate(frames):
            filename = frame[0]
            label = frame[1]

            # Get the image path.
            image = image_path + filename

            # Read in the image_data
            # image_data = tf.gfile.FastGFile(image, 'rb').read()
            t = read_tensor_from_image_file(image)

            # graph init support 
            input_name = "import/Placeholder"
            output_name = "import/final_result"
            input_operation = graph.get_operation_by_name(input_name)
            output_operation = graph.get_operation_by_name(output_name)

            try:
                # predictions = sess.run(
                #     softmax_tensor,
                #     {'DecodeJpeg/contents:0': image_data}
                # )
                predictions = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
                predictions = np.squeeze(predictions)
                prediction = predictions[0]
            except KeyboardInterrupt:
                print("You quit with ctrl+c")
                sys.exit()
            except:
                print("Error making prediction, continuing.")
                continue

            # Save the probability that it's each of our classes.
            frame_predictions.append([prediction, label])

            if i > 0 and i % 10 == 0:
                pbar.update(10)

        pbar.close()

        return frame_predictions

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def get_accuracy(predictions, labels):
    """After predicting on each batch, check that batch's
    accuracy to make sure things are good to go. This is
    a simple accuracy metric, and so doesn't take confidence
    into account, which would be a better metric to use to
    compare changes in the model."""
    correct = 0
    for frame in predictions:
        # Get the highest confidence class.
        this_prediction = frame[0].tolist()
        this_label = frame[1]

        max_value = max(this_prediction)
        max_index = this_prediction.index(max_value)
        predicted_label = labels[max_index]

        # Now see if it matches.
        if predicted_label == this_label:
            correct += 1

    accuracy = correct / len(predictions)
    return accuracy

def main():
    batches = ['1']
    labels = get_labels()

    for batch in batches:
        print("Doing batch %s" % batch)
        with open('data/labeled-frames.pkl', 'rb') as fin:
            frames = pickle.load(fin)

        # Predict on this batch and get the accuracy.
        predictions = predict_on_frames(frames, batch)
        accuracy = get_accuracy(predictions, labels)
        print("Batch accuracy: %.5f" % accuracy)

        # Save it.
        with open('data/predicted-frames.pkl', 'wb') as fout:
            pickle.dump(predictions, fout)

    print("Done.")

if __name__ == '__main__':
    main()

