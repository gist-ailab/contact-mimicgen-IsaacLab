# read hdf5 file
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2


def load_dataset_helper(data_group):
    data = {}
    for key in data_group:
        if isinstance(data_group[key], h5py.Group):
            data[key] = load_dataset_helper(data_group[key])
        else:
            data[key] = torch.tensor(np.array(data_group[key]))
    
    return data

def visualize_camera_data(data, demo_idx=0, max_frames=10):
    """카메라 데이터를 시각화하는 함수"""
    print("=== 카메라 데이터 구조 확인 ===")
    
    # 데이터 구조 탐색
    def explore_structure(data, prefix=""):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    print(f"{prefix}{key}/")
                    explore_structure(value, prefix + "  ")
                elif isinstance(value, torch.Tensor):
                    print(f"{prefix}{key}: {value.shape} {value.dtype}")
                else:
                    print(f"{prefix}{key}: {type(value)}")
        else:
            print(f"{prefix}{type(data)}")
    
    explore_structure(data)
    
    # 카메라 데이터 찾기
    camera_data = {}
    
    def find_camera_data(data, prefix=""):
        if isinstance(data, dict):
            for key, value in data.items():
                if "*_cam" in key or "cam" in key.lower():
                    camera_data[f"{prefix}{key}"] = value
                    print(f"카메라 데이터 발견: {prefix}{key} - {value.shape if hasattr(value, 'shape') else type(value)}")
                elif isinstance(value, dict):
                    find_camera_data(value, f"{prefix}{key}/")
                elif isinstance(value, torch.Tensor) and len(value.shape) >= 3:
                    # 이미지 형태의 텐서인지 확인 (H, W, C 또는 C, H, W)
                    if len(value.shape) == 3 and (value.shape[-1] == 3 or value.shape[0] == 3):
                        camera_data[f"{prefix}{key}"] = value
                        print(f"이미지 데이터 발견: {prefix}{key} - {value.shape}")
        elif isinstance(data, torch.Tensor) and len(data.shape) >= 3:
            if len(data.shape) == 3 and (data.shape[-1] == 3 or data.shape[0] == 3):
                camera_data[prefix] = data
                print(f"이미지 데이터 발견: {prefix} - {data.shape}")
    
    find_camera_data(data)
    
    # 카메라 데이터 시각화
    if camera_data:
        print(f"\n=== {len(camera_data)}개의 카메라/이미지 데이터 발견 ===")
        
        for cam_name, cam_data in camera_data.items():
            print(f"\n카메라: {cam_name}")
            print(f"데이터 형태: {cam_data.shape}")
            
            if len(cam_data.shape) == 4:  # (T, H, W, C) 또는 (T, C, H, W)
                # 시퀀스 데이터인 경우
                num_frames = min(cam_data.shape[0], max_frames)
                print(f"시퀀스 길이: {cam_data.shape[0]}, 표시할 프레임: {num_frames}")
                
                # 첫 번째 프레임 시각화
                if cam_data.shape[-1] == 3:  # (T, H, W, C)
                    first_frame = cam_data[0].numpy()
                else:  # (T, C, H, W)
                    first_frame = cam_data[0].permute(1, 2, 0).numpy()
                
                plt.figure(figsize=(10, 6))
                plt.imshow(first_frame.astype(np.uint8))
                plt.title(f"{cam_name} - Frame 0")
                plt.axis('off')
                plt.show()
                
                # 여러 프레임을 그리드로 표시
                if num_frames > 1:
                    fig, axes = plt.subplots(2, min(5, num_frames), figsize=(15, 6))
                    if num_frames == 1:
                        axes = [axes]
                    elif len(axes.shape) == 1:
                        axes = axes.reshape(1, -1)
                    
                    for i in range(min(10, num_frames)):
                        row = i // 5
                        col = i % 5
                        
                        if cam_data.shape[-1] == 3:  # (T, H, W, C)
                            frame = cam_data[i].numpy()
                        else:  # (T, C, H, W)
                            frame = cam_data[i].permute(1, 2, 0).numpy()
                        
                        axes[row, col].imshow(frame.astype(np.uint8))
                        axes[row, col].set_title(f"Frame {i}")
                        axes[row, col].axis('off')
                    
                    # 빈 subplot 숨기기
                    for i in range(num_frames, axes.size):
                        row = i // 5
                        col = i % 5
                        axes[row, col].axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                    
            elif len(cam_data.shape) == 3:  # (H, W, C) 또는 (C, H, W)
                # 단일 이미지
                if cam_data.shape[-1] == 3:  # (H, W, C)
                    image = cam_data.numpy()
                else:  # (C, H, W)
                    image = cam_data.permute(1, 2, 0).numpy()
                
                plt.figure(figsize=(8, 6))
                plt.imshow(image.astype(np.uint8))
                plt.title(f"{cam_name}")
                plt.axis('off')
                plt.show()
                
    else:
        print("카메라 데이터를 찾을 수 없습니다.")
        print("데이터 구조를 더 자세히 확인해보겠습니다...")
        
        # obs 데이터 내부 확인
        if 'obs' in data:
            print("\n=== obs 데이터 구조 ===")
            explore_structure(data['obs'])
            
            # obs 내부에서 카메라 데이터 찾기
            if isinstance(data['obs'], dict):
                for key, value in data['obs'].items():
                    if isinstance(value, torch.Tensor) and len(value.shape) >= 3:
                        print(f"obs.{key}: {value.shape}")
                        if len(value.shape) == 4:  # 시퀀스 데이터
                            print(f"  시퀀스 길이: {value.shape[0]}")
                            print(f"  이미지 크기: {value.shape[1:] if value.shape[-1] == 3 else value.shape[2:]}")
                        elif len(value.shape) == 3:  # 단일 이미지
                            print(f"  이미지 크기: {value.shape}")

# read hdf5 file
with h5py.File('datasets/test_dataset.hdf5', 'r') as f:
    print("=== HDF5 파일 구조 ===")
    print("Top-level keys:", list(f.keys()))
    
    # 데이터 그룹 확인
    if 'data' in f:
        print("\n=== 데이터 그룹 ===")
        data_group = f['data']
        print("Demo keys:", list(data_group.keys()))
        
        # 첫 번째 데모 데이터 로드
        demo_key = list(data_group.keys())[0]
        print(f"\n=== {demo_key} 데이터 로드 ===")
        data = load_dataset_helper(data_group[demo_key])
        
        # 카메라 데이터 시각화
        visualize_camera_data(data)
        
    else:
        print("'data' 그룹을 찾을 수 없습니다.")
        print("사용 가능한 그룹:", list(f.keys()))
