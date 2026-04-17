# RobotStatus.msg - Robot agent status heartbeat
builtin_interfaces/Time stamp
string robot_id

# Status constants
uint8 STATUS_IDLE = 0
uint8 STATUS_TRAINING = 1
uint8 STATUS_UPLOADING = 2
uint8 STATUS_WAITING = 3
uint8 STATUS_ERROR = 4

uint8 status
int32 current_round
int32 total_rounds_participated
float32 last_loss
float32 last_accuracy
int32 model_parameter_count
float32 cpu_usage_percent
float32 memory_usage_mb
bool is_active
