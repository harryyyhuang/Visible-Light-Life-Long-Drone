#ifndef CONTROLLER_PID_H
#define CONTROLLER_PID_H

#ifdef __cplusplus
extern "C" {
#endif

// --- Structs for control, setpoint, state, and sensors ---
typedef struct {
    float v[3]; // x, y, z or roll, pitch, yaw
} vec3f;

typedef struct {
    vec3f position;
    vec3f velocity;
    vec3f attitude;      // roll, pitch, yaw (rad)
    vec3f attitudeRate;  // roll, pitch, yaw rates (rad/s)
} setpoint_t;

typedef struct {
    vec3f position;
    vec3f velocity;
    vec3f attitude;      // roll, pitch, yaw (rad)
    vec3f omega;         // roll, pitch, yaw rates (rad/s)
} state_t;

typedef struct {
    vec3f gyro; // x, y, z (rad/s)
    vec3f acc;  // x, y, z (m/s^2)
} sensorData_t;

typedef struct {
    float thrust;
    float roll;
    float pitch;
    float yaw;
} control_t;

// --- Extern gain/limit variables for tuning ---
extern float Kp_pos[3], Ki_pos[3], Kd_pos[3], pos_integral_limit[3];
extern float Kp_att[3], Ki_att[3], Kd_att[3], att_integral_limit[3];
extern float Kp_rate[3], Ki_rate[3], Kd_rate[3], rate_integral_limit[3];
extern float dt;

// --- Main PID controller function ---
void controllerPid(control_t *control, const setpoint_t *setpoint, const sensorData_t *sensors, 
                    const state_t *state, float accumulatedTime);
void controllerPidReset();

#ifdef __cplusplus
}
#endif

#endif // CONTROLLER_PID_H
