""" Some code to help with Assignment 1. """

def reduce_dimensions(wv):
    """ Reduces dimensions of word vectors (as obtained from gensim) to 2 dimensions.
        Returns the values in the two dimensions (x_vals & y_vals) in numpy arrays,
        and the list of corresponding words(labels).
        Adapted from the Gensim's tutorial. """
    from sklearn.manifold import TSNE                   # for dimensionality reduction
    import numpy as np                                  # array handling

    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(wv.vectors)
    labels = wv.index_to_key  # list

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels

def plot_with_matplotlib(x_vals, y_vals, labels, words_to_plot):
    """ A function to visualize word embeddings reduced to two dimensions.
        x_vals, y_vals and labels are as returned by the reduce_dimensions function.
        words_to_plot is a list of words.
        Adapted from the Gensim's tutorial. """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    for w in words_to_plot:
        if w in labels :
            i = labels.index(w)
            print("Plotting",w,"at",x_vals[i], y_vals[i])
            plt.annotate(w, (x_vals[i], y_vals[i]))
        else :
            print(w,"cannot be plotted because its word embedding is not given.")
    plt.show()
