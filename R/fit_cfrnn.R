#' CF-RNN Forecasting Model
#'
#' Fits a Vanilla recurrent neural network for multi-horizon forecasting
#' using multiple independent time series.
#'
#' The independent time series are first divided into reference and test
#' sets. The reference set is further divided into training and calibration
#' sets. Only the training set is used to fit the RNN.
#'
#' @param data A numeric matrix where rows correspond to independent time
#'   series. The first T columns contain observed values and the final H
#'   columns contain future values.
#' @param H A positive integer specifying the number of future forecasting
#'   horizons.
#' @param ref_prop Proportion of independent series assigned to the reference
#'   set.
#' @param train_prop Proportion of the reference set assigned to training.
#' @param hidden_size Number of hidden units in the Vanilla RNN.
#' @param epochs Number of training epochs.
#' @param batch_size Mini-batch size used during training.
#' @param learning_rate Learning rate used by the Adam optimizer.
#' @param seed Optional random seed used for splitting the independent series.
#'
#' @return A list containing the fitted RNN model, training/calibration/test
#'   splits, and model settings.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' data <- matrix(rnorm(100 * 25), nrow = 100)
#'
#' fit <- fit_cfrnn(
#'   data = data,
#'   H = 5,
#'   seed = 123
#' )
#' }
fit_cfrnn <- function(data,
                      H,
                      ref_prop = 0.8,
                      train_prop = 0.75,
                      hidden_size = 32,
                      epochs = 100,
                      batch_size = 32,
                      learning_rate = 0.001,
                      seed = NULL) {

  # Split independent series into train, calibration, and test sets
  split <- split_cfrnn_data(
    data = data,
    H = H,
    ref_prop = ref_prop,
    train_prop = train_prop,
    seed = seed
  )

  train_x <- split$train_x
  train_y <- split$train_y

  # Convert training inputs to torch tensors
  X_tensor <- torch::torch_tensor(
    train_x,
    dtype = torch::torch_float()
  )

  Y_tensor <- torch::torch_tensor(
    train_y,
    dtype = torch::torch_float()
  )

  # Shape:
  # l_train x T  -->  l_train x T x 1
  X_tensor <- X_tensor$unsqueeze(3)

  # Initialize RNN model
  model <- cfrnn_model(
    hidden_size = hidden_size,
    H = H
  )

  optimizer <- torch::optim_adam(
    model$parameters,
    lr = learning_rate
  )

  n_train <- nrow(train_x)

  loss_history <- numeric(epochs)

  # Training

  for (epoch in seq_len(epochs)) {

    model$train()

    # Shuffle training series at each epoch
    indices <- sample(seq_len(n_train))

    epoch_loss <- 0
    n_batches <- 0

    batch_starts <- seq(1, n_train, by = batch_size)

    for (start in batch_starts) {

      end <- min(start + batch_size - 1, n_train)

      batch_index <- indices[start:end]

      x_batch <- X_tensor[batch_index, , ]
      y_batch <- Y_tensor[batch_index, ]

      optimizer$zero_grad()

      predictions <- model(x_batch)

      loss <- torch::nnf_mse_loss(
        predictions,
        y_batch
      )

      loss$backward()

      optimizer$step()

      epoch_loss <- epoch_loss + loss$item()
      n_batches <- n_batches + 1
    }

    loss_history[epoch] <- epoch_loss / n_batches
  }

  result <- list(
    model = model,

    train_x = split$train_x,
    train_y = split$train_y,

    calib_x = split$calib_x,
    calib_y = split$calib_y,

    test_x = split$test_x,
    test_y = split$test_y,

    T = split$T,
    H = H,

    ref_prop = ref_prop,
    train_prop = train_prop,

    hidden_size = hidden_size,
    epochs = epochs,
    batch_size = batch_size,
    learning_rate = learning_rate,

    training_loss = loss_history
  )

  class(result) <- "cfrnn_fit"

  return(result)
}
