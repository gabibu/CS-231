import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):

    scores = X[i].dot(W)
    y_score = scores[y[i]]
    max_score = np.max(scores)
    scores -= max_score

    scores_exponents = np.exp(scores)
    scores_exponents_sum = scores_exponents.sum()

    loss += -y_score + max_score + np.log(scores_exponents_sum)

    for j in range(num_classes):
      dW[:, j] += np.exp(scores[j]) / scores_exponents_sum * X[i, :]

    dW[:, y[i]] -= X[i, :]

  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += reg * np.sum(W * W)


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_train = X.shape[0]
  scores = X.dot(W)

  y_scores = np.take_along_axis(scores, y[:, None], axis=1).flatten()

  max_scores = np.max(scores, axis=1, keepdims=True)
  scores -= max_scores

  scores_exponents = np.exp(scores)


  loss = -y_scores.sum() + max_scores.sum() + np.log(np.exp(scores).sum(axis=1)).sum()

  loss /= num_train
  loss += reg * np.sum(W * W)

  softmax_deriv = (scores_exponents / scores_exponents.sum(axis=1).reshape(-1, 1))

  set_y_softmax_deriv_values = np.take_along_axis(softmax_deriv, y[:, None], axis=1) - 1

  np.put_along_axis(softmax_deriv, y[:, None], set_y_softmax_deriv_values, axis=1)


  dW = X.T.dot(softmax_deriv)
  dW /= num_train
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

