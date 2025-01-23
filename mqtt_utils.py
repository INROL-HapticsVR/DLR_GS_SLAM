import paho.mqtt.client as mqtt
import numpy as np
import os
import json
import struct

# MQTT 설정5
MQTT_BROKER = "147.46.149.20"
# MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC = "topic/gaussian"

# MQTT 클라이언트 생성
mqtt_client = mqtt.Client()

def connect_mqtt():
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
    print("Connected to MQTT broker.")


def serialize_gaussian_to_binary(global_index_send, new_xyz, new_colors_rgba, new_scales, new_rots, new_ids):
    """
    가우시안 데이터를 바이너리로 직렬화, 총 갯수 포함
    """
    binary_data = bytearray()

    # global_index_send를 uint32로 맨 앞에 추가
    binary_data.extend(struct.pack('I', global_index_send))

    # 총 갯수 추가 (global_index_send 다음에 저장)
    total_count = len(new_ids)
    binary_data.extend(struct.pack('i', total_count))

    # 데이터를 float32 형식으로 고정
    new_ids = np.array(new_ids, dtype=np.uint32)  # ID는 uint32
    new_xyz = np.array(new_xyz, dtype=np.float32)  # Position (3 floats)
    new_colors_rgba = np.array(new_colors_rgba, dtype=np.float32)  # Color (RGBA, 4 floats)
    new_scales = np.array(new_scales, dtype=np.float32)  # Scale (3 floats)
    new_rots = np.array(new_rots, dtype=np.float32)  # Rotation (4 floats)

    for i in range(total_count):
        # struct.pack으로 데이터를 바이너리로 직렬화
        binary_data.extend(
            struct.pack(
                'I3f4f3f4f',  # ID (uint32), 3x pos (float), 4x color (float), 3x scale (float), 4x rotation (float)
                int(new_ids[i]),            # ID (uint32)
                *new_xyz[i].astype(np.float32),    # Position
                *new_colors_rgba[i].astype(np.float32),  # Color (RGBA)
                *new_scales[i].astype(np.float32),       # Scale
                *new_rots[i].astype(np.float32)          # Rotation
            )
        )
    return binary_data


def save_binary_to_file(filename, binary_data):
    """
    바이너리 데이터를 파일로 저장
    """
    with open(filename, "wb") as f:
        f.write(binary_data)
    print(f"Binary data saved to {filename}")


def serialize_gaussian_to_json(new_xyz, new_colors_rgba, new_scales, new_rots, new_ids):
    """
    가우시안 데이터를 JSON으로 직렬화 (ID 포함)
    """
    data = {
        "gaussians": []
    }

    for i in range(new_xyz.shape[0]):
        gaussian = {
            "id": int(new_ids[i]),  # ID 추가
            "pos": new_xyz[i].tolist(),
            "color": new_colors_rgba[i].tolist(),
            "scale": new_scales[i].tolist(),
            "rotation": new_rots[i].tolist()
        }
        data["gaussians"].append(gaussian)

    return json.dumps(data)


def save_json_to_file(filename, serialized_data):
    """
    가우시안 데이터를 JSON 파일에 저장.
    기존 파일이 있으면 데이터를 병합하고 없으면 새로 생성.

    Args:
        filename (str): 저장할 JSON 파일 이름.
        serialized_data (str): JSON 직렬화된 가우시안 데이터.
    """
    # 직렬화된 데이터를 Python 객체로 변환
    new_data = json.loads(serialized_data)

    # 기존 JSON 파일 읽기
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {"gaussians": []}
    else:
        existing_data = {"gaussians": []}

    # 새로 추가된 가우시안 데이터 병합
    if "gaussians" in new_data:
        existing_data["gaussians"].extend(new_data["gaussians"])

    # 업데이트된 데이터를 다시 저장
    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=4)

    print(f"Appended Gaussian data to {filename}")



def send_to_mqtt(topic, serialized_data):
    """
    MQTT로 데이터 전송
    """
    mqtt_client.publish(topic, serialized_data)
    print(f"Published data to {topic}")


def sendTest_to_mqtt(topic):
    """
    MQTT로 데이터 전송
    """
    mqtt_client.publish(topic, 'aaa')
    print(f"Published data to {topic}")