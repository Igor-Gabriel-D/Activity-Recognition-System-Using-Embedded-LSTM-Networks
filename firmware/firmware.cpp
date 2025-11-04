#include <stdio.h>
#include <cmath>
#include <cstring>
#include "pico/stdlib.h"
#include "hardware/i2c.h"
#include "pico/cyw43_arch.h"
#include "lwip/apps/mqtt.h"
#include "lwip/ip_addr.h"
#include "pico/multicore.h"

#include "include/standard_scaler.h"
#include "include/x_test_pico.h"
#include "include/model.h"

//#define WIFI_SSID "Gabriel"
//#define WIFI_PASS "88458737"

#define WIFI_SSID "tarefa-mqtt"
#define WIFI_PASS "laica@2025"

#define I2C_PORT i2c1
#define SDA_PIN 2
#define SCL_PIN 3

#define MPU6050_ADDR 0x68
#define REG_PWR_MGMT_1 0x6B
#define REG_ACCEL_XOUT_H 0x3B
#define REG_GYRO_XOUT_H  0x43
#define REG_ACCEL_CONFIG 0x1C

#define TIME_STEPS 30
#define INPUT_SIZE 1
#define UNITS1 32
#define UNITS2 32
#define UNITS3 32
#define OUTPUT_SIZE 4


#define MQTT_SERVER_IP "54.36.178.49" // test.mosquitto.org
#define MQTT_TOPIC "tcc/igor"
#define MQTT_RESULT_TOPIC "tcc/igor/result"

mqtt_client_t *client;

void mqtt_pub_request_cb(void *arg, err_t err) {
    printf("Publicação concluída, status = %d\n", err);
}

// Callback de subscribe
void mqtt_sub_request_cb(void *arg, err_t err) {
    printf("Inscrição concluída, status = %d\n", err);
}

// Callback de mensagens recebidas (meta)
void mqtt_incoming_publish_cb(void *arg, const char *topic, u32_t tot_len) {
    printf("Mensagem recebida no tópico %s, tamanho %lu\n", topic, tot_len);
}

// Callback de dados recebidos — aqui reagimos à mensagem e executamos predição
void mqtt_incoming_data_cb(void *arg, const u8_t *data, u16_t len, u8_t flags) {
    // Copia payload para string terminada em '\0'
    char buf[1024];
    if (len >= sizeof(buf)) len = sizeof(buf) - 1;
    memcpy(buf, data, len);
    buf[len] = '\0';

    printf("Payload: %s\n", buf);

    static float input_seq[TIME_STEPS][INPUT_SIZE]; // buffer reutilizável

    // Só aceita comandos "predict" ou "run"
    if (strcmp(buf, "predict") == 0 || strcmp(buf, "run") == 0) {
        printf("Trigger de predição recebido (usando input_seq default)\n");
        
    } else {
        printf("Comando inválido. Use apenas 'predict' ou 'run'.\n");
    }
}

// Callback de conexão MQTT
void mqtt_connection_cb(mqtt_client_t *client, void *arg, mqtt_connection_status_t status) {
    if (status == MQTT_CONNECT_ACCEPTED) {
        printf("MQTT conectado!\n");

        // Configura callbacks de recebimento
        mqtt_set_inpub_callback(client, mqtt_incoming_publish_cb, mqtt_incoming_data_cb, NULL);

        // Assina tópico
        mqtt_subscribe(client, MQTT_TOPIC, 0, mqtt_sub_request_cb, NULL);

        // Publica mensagem de boas-vindas
        const char *msg = "Hello MQTT!";
        mqtt_publish(client, MQTT_TOPIC, msg, strlen(msg), 0, 0, mqtt_pub_request_cb, NULL);

    } else {
        printf("Falha na conexão MQTT, status = %d\n", status);
    }
}

int16_t read_word(uint8_t reg) {
    uint8_t buf[2];
    i2c_write_blocking(I2C_PORT, MPU6050_ADDR, &reg, 1, true);
    i2c_read_blocking(I2C_PORT, MPU6050_ADDR, buf, 2, false);
    return (int16_t)((buf[0] << 8) | buf[1]);
}

void mpu6050_wake() {
    uint8_t buf[2] = {REG_PWR_MGMT_1, 0x00};
    i2c_write_blocking(I2C_PORT, MPU6050_ADDR, buf, 2, false);
}

void mpu6050_set_accel_range(uint8_t range) {
    // range = 0: ±2g, 1: ±4g, 2: ±8g, 3: ±16g
    uint8_t value = range << 3;
    uint8_t buf[2] = {REG_ACCEL_CONFIG, value};
    i2c_write_blocking(I2C_PORT, MPU6050_ADDR, buf, 2, false);
}

void mpu6050_read_accel(float *ax, float *ay, float *az, uint8_t range) {
    float modifier;
    switch (range) {
        case 0: modifier = 16384.0; break; // +-2
        case 1: modifier = 8192.0; break;
        case 2: modifier = 4096.0; break;
        case 3: modifier = 2048.0; break;
        default: modifier = 16384.0; break;
    }

    int16_t raw_ax = read_word(REG_ACCEL_XOUT_H);
    int16_t raw_ay = read_word(REG_ACCEL_XOUT_H + 2);
    int16_t raw_az = read_word(REG_ACCEL_XOUT_H + 4);

    *ax = raw_ax / modifier;
    *ay = raw_ay / modifier;
    *az = raw_az / modifier;
}

// --- Funções auxiliares ---
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float tanh_custom(float x) {
    return tanhf(x);
}

void matmul(const float* vec, int vec_size,
            const float* mat, int mat_rows, int mat_cols,
            float* out) {
    for (int j = 0; j < mat_cols; j++) {
        float acc = 0.0f;
        for (int i = 0; i < vec_size; i++)
            acc += vec[i] * mat[i * mat_cols + j];
        out[j] = acc;
    }
}

void lstm_forward_step(const float* input, int input_size,
                       const float* h_prev, const float* c_prev, int units,
                       const float* kernel, const float* recurrent_kernel, const float* bias,
                       float* h_t, float* c_t) {
    float gates[4 * units];
    float x_proj[4 * units];
    float h_proj[4 * units];

    matmul(input, input_size, kernel, input_size, 4 * units, x_proj);
    matmul(h_prev, units, recurrent_kernel, units, 4 * units, h_proj);

    for (int i = 0; i < 4 * units; i++)
        gates[i] = x_proj[i] + h_proj[i] + bias[i];

    for (int i = 0; i < units; i++) {
        float i_gate = sigmoid(gates[i]);
        float f_gate = sigmoid(gates[i + units]);
        float c_gate = tanh_custom(gates[i + 2 * units]);
        float o_gate = sigmoid(gates[i + 3 * units]);

        c_t[i] = f_gate * c_prev[i] + i_gate * c_gate;
        h_t[i] = o_gate * tanh_custom(c_t[i]);
    }
}

void dense_softmax(const float* input, int in_size,
                   const float* kernel, const float* bias, int out_size,
                   float* out) {
    for (int j = 0; j < out_size; j++) {
        float acc = bias[j];
        for (int i = 0; i < in_size; i++)
            acc += input[i] * kernel[i * out_size + j];
        out[j] = acc;
    }

    // Softmax estável
    float maxv = out[0];
    for (int i = 1; i < out_size; i++)
        if (out[i] > maxv) maxv = out[i];

    float sum = 0.0f;
    for (int i = 0; i < out_size; i++)
        sum += expf(out[i] - maxv);

    for (int i = 0; i < out_size; i++)
        out[i] = expf(out[i] - maxv) / sum;
}

void run_prediction(float input_seq[TIME_STEPS][INPUT_SIZE], int sample_idx) {
    float h1_prev[UNITS1] = {0}, c1_prev[UNITS1] = {0}, h1_t[UNITS1], c1_t[UNITS1];
    static float seq_out1[TIME_STEPS][UNITS1];

    for (int t = 0; t < TIME_STEPS; t++) {
        lstm_forward_step(input_seq[t], INPUT_SIZE, h1_prev, c1_prev, UNITS1,
                          lstm1_kernel, lstm1_recurrent_kernel, lstm1_bias, h1_t, c1_t);
        memcpy(h1_prev, h1_t, sizeof(h1_t));
        memcpy(c1_prev, c1_t, sizeof(c1_t));
        memcpy(seq_out1[t], h1_t, sizeof(h1_t));
    }

    float h2_prev[UNITS2] = {0}, c2_prev[UNITS2] = {0}, h2_t[UNITS2], c2_t[UNITS2];
    static float seq_out2[TIME_STEPS][UNITS2];

    for (int t = 0; t < TIME_STEPS; t++) {
        lstm_forward_step(seq_out1[t], UNITS1, h2_prev, c2_prev, UNITS2,
                          lstm2_kernel, lstm2_recurrent_kernel, lstm2_bias, h2_t, c2_t);
        memcpy(h2_prev, h2_t, sizeof(h2_t));
        memcpy(c2_prev, c2_t, sizeof(c2_t));
        memcpy(seq_out2[t], h2_t, sizeof(h2_t));
    }

    float h3_prev[UNITS3] = {0}, c3_prev[UNITS3] = {0}, h3_t[UNITS3], c3_t[UNITS3];
    for (int t = 0; t < TIME_STEPS; t++) {
        lstm_forward_step(seq_out2[t], UNITS2, h3_prev, c3_prev, UNITS3,
                          lstm3_kernel, lstm3_recurrent_kernel, lstm3_bias, h3_t, c3_t);
        memcpy(h3_prev, h3_t, sizeof(h3_t));
        memcpy(c3_prev, c3_t, sizeof(c3_t));
    }

    float probs[OUTPUT_SIZE];
    dense_softmax(h3_t, UNITS3, dense_kernel, dense_bias, OUTPUT_SIZE, probs);

    int pred_class = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++)
        if (probs[i] > probs[pred_class])
            pred_class = i;

    printf("\n===== Amostra %d =====\n", sample_idx + 1);
    for (int i = 0; i < OUTPUT_SIZE; i++)
        printf("Classe %d: %.3f\n", i, probs[i]);
    printf("Classe predita: %d\n", pred_class);
}



// Função para aplicar o Standard Scaler
inline void standard_scale(float* input, float* output, size_t length) {
    const size_t scaler_len = sizeof(scaler_mean) / sizeof(scaler_mean[0]);
    if (length != scaler_len) {
        
        return;
    }
    
    for (size_t i = 0; i < length; i++) {
        output[i] = (input[i] - scaler_mean[i]) / scaler_scale[i];
    }
}

#define QUEUE_LEN 1
float shared_input_seq[TIME_STEPS][INPUT_SIZE];
bool ready_for_prediction = false;

void core1_entry() {
    while (true) {
        if (ready_for_prediction) {
            run_prediction(shared_input_seq, 0);
            ready_for_prediction = false;
        }
        sleep_ms(1);
    }
}

int main()
{
    stdio_init_all();
    multicore_launch_core1(core1_entry);

        //==================================================================================================
    // Inicializa Wi-Fi
    if (cyw43_arch_init()) {
        printf("Falha ao inicializar Wi-Fi\n");
        return -1;
    }

    cyw43_arch_enable_sta_mode();

    if (cyw43_arch_wifi_connect_timeout_ms(WIFI_SSID, WIFI_PASS, CYW43_AUTH_WPA2_AES_PSK, 30000)) {
        printf("Falha ao conectar Wi-Fi\n");
        return -1;
    }

    printf("Wi-Fi conectado!\n");

    // Cria cliente MQTT
    client = mqtt_client_new();
    if (!client) {
        printf("Erro ao criar cliente MQTT\n");
        return -1;
    }

    // Configura informações do cliente
    struct mqtt_connect_client_info_t ci = {
        .client_id = "pico_w",
        .client_user = NULL,
        .client_pass = NULL,
        .keep_alive = 60,
        .will_topic = NULL,
        .will_msg = NULL,
        .will_msg_len = 0,
        .will_qos = 0,
        .will_retain = 0
    };

    ip_addr_t server_ip;
    ipaddr_aton(MQTT_SERVER_IP, &server_ip);

    // Conecta ao broker MQTT
    mqtt_client_connect(client, &server_ip, MQTT_PORT, mqtt_connection_cb, NULL, &ci);

    int index = 0;
    float ax, ay, az;
    
    i2c_init(I2C_PORT, 100 * 1000);
    gpio_set_function(SDA_PIN, GPIO_FUNC_I2C);
    gpio_set_function(SCL_PIN, GPIO_FUNC_I2C);
    gpio_pull_up(SDA_PIN);
    gpio_pull_up(SCL_PIN);
    
    const size_t scaler_len = sizeof(scaler_mean) / sizeof(scaler_mean[0]);
    const size_t total_len = sizeof(X_test_pico) / sizeof(X_test_pico[0]);
    const size_t num_samples = total_len / scaler_len;
    
    float* scaled = new float[total_len];
    
    for (size_t i = 0; i < num_samples; i++) {
        float* input_ptr  = &X_test_pico[i * scaler_len];
        float* output_ptr = &scaled[i * scaler_len];
        standard_scale(input_ptr, output_ptr, scaler_len);
    }
    sleep_ms(4000);

    size_t sample_idx = 3;  // 0 = primeira, 1 = segunda, 2 = terceira, etc.

    for (size_t j = 0; j < scaler_len; j++) {
        printf("%.3f ",scaled[sample_idx * scaler_len + j]);
    }
    printf("\n");

        // --- Prepara entrada para o modelo [TIME_STEPS][INPUT_SIZE] ---
    float input_seq[TIME_STEPS][INPUT_SIZE];
    for (int t = 0; t < TIME_STEPS; t++) {
        input_seq[t][0] = scaled[sample_idx * scaler_len + t];  // INPUT_SIZE = 1
    }

    // --- Executa o modelo ---
    run_prediction(input_seq, sample_idx);

    sleep_ms(10000);

    #define WINDOW_SIZE 10 // 10 leituras por eixo
    float buffer_x[WINDOW_SIZE] = {0};
    float buffer_y[WINDOW_SIZE] = {0};
    float buffer_z[WINDOW_SIZE] = {0};
    int total_samples = 0;
    while (true) {
        


        mpu6050_read_accel(&ax, &ay, &az, 0);

        // Imprime os valores lidos
        printf("%d, %.3f, %.3f, %.3f\n", index++, ax, ay, az);

            // --- desloca as amostras antigas ---
        for (int i = 0; i < WINDOW_SIZE - 1; i++) {
            buffer_x[i] = buffer_x[i + 1];
            buffer_y[i] = buffer_y[i + 1];
            buffer_z[i] = buffer_z[i + 1];
        }

        // adiciona nova amostra no final
        buffer_x[WINDOW_SIZE - 1] = ax;
        buffer_y[WINDOW_SIZE - 1] = ay;
        buffer_z[WINDOW_SIZE - 1] = az;

        // aguarda encher a primeira janela
        if (total_samples < WINDOW_SIZE) {
            total_samples++;
            sleep_ms(200);
            continue;
        }

        // --- Aplica StandardScaler ---
        float scaled_seq[30]; // 10 X + 10 Y + 10 Z
        for (int i = 0; i < WINDOW_SIZE; i++) {
            scaled_seq[i]       = (buffer_x[i] - scaler_mean[i]) / scaler_scale[i];
            scaled_seq[i + 10]  = (buffer_y[i] - scaler_mean[i + 10]) / scaler_scale[i + 10];
            scaled_seq[i + 20]  = (buffer_z[i] - scaler_mean[i + 20]) / scaler_scale[i + 20];
        }

        // --- Monta input_seq ---
        float input_seq[TIME_STEPS][INPUT_SIZE];
        for (int t = 0; t < TIME_STEPS; t++) {
            input_seq[t][0] = scaled_seq[t];
        }


        for (int t = 0; t < TIME_STEPS; t++) {
            shared_input_seq[t][0] = scaled_seq[t];
        }

        // Sinaliza que há nova sequência pra predição
        ready_for_prediction = true;



        sleep_ms(200);
    }
}
