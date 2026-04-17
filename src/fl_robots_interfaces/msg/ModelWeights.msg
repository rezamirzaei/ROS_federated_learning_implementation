# ModelWeights.msg - Serialized neural network model weights
# Used for transmitting model parameters between robots and aggregator

builtin_interfaces/Time stamp
string robot_id                  # Source robot identifier
int32 round_number               # Training round this belongs to
string[] layer_names             # Names of each layer
float32[] flat_weights           # All weights concatenated into flat array
int32[] layer_sizes              # Size of each layer's weights (for reconstruction)
int32 total_samples              # Number of samples used in training
float32 training_loss            # Loss after local training
float32 training_accuracy        # Accuracy after local training
