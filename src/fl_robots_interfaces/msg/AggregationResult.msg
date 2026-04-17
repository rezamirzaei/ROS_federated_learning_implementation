# AggregationResult.msg - Result of a federated averaging round
builtin_interfaces/Time stamp
int32 round_number
int32 num_participants
int32 total_samples
float32 aggregation_time_ms
float32 mean_divergence
float32 max_divergence
string[] participant_ids
float32[] participant_losses
float32[] participant_accuracies
