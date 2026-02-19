#!/usr/bin/env python3
import numpy as np
import struct
from typing import Tuple, Set

# ----------------------------
# Helpers
# ----------------------------
def conv_elems(inC: int, outC: int, kH: int, kW: int) -> int:
    return inC * outC * kH * kW

def write_binary_layer(fh, inC: int, outC: int, kH: int, kW: int) -> int:
    n_conv = conv_elems(inC, outC, kH, kW)
    n_bn   = outC
    conv_w = np.random.choice([-1.0, 1.0], size=n_conv).astype(np.float32)
    bn     = np.random.uniform(-2.0, 2.0, size=n_bn).astype(np.float32)
    for x in conv_w: fh.write(f"{x}\n")
    for x in bn:     fh.write(f"{x}\n")
    return n_conv + n_bn

def write_ternary_layer(fh, inC: int, outC: int, kH: int, kW: int, mutually_exclusive=True) -> int:
    n_plane = conv_elems(inC, outC, kH, kW)
    n_bn    = outC
    if mutually_exclusive:
        # True ternary {-1,0,+1} -> two planes in {0,1}
        T = np.random.choice([-1.0, 0.0, 1.0], size=n_plane).astype(np.float32)
        pos = (T > 0).astype(np.float32)
        neg = (T < 0).astype(np.float32)
    else:
        # Independent planes in {0,1} (still valid as w = pos - neg)
        pos = np.random.choice([0.0, 1.0], size=n_plane).astype(np.float32)
        neg = np.random.choice([0.0, 1.0], size=n_plane).astype(np.float32)

    bn = np.random.uniform(-2.0, 2.0, size=n_bn).astype(np.float32)
    for x in pos: fh.write(f"{x}\n")
    for x in neg: fh.write(f"{x}\n")
    for x in bn:  fh.write(f"{x}\n")
    return n_plane + n_plane + n_bn

def write_inconv1(fh, inC=3, outC=64, kH=3, kW=3) -> int:
    n_conv = conv_elems(inC, outC, kH, kW)
    n_bn   = 2 * outC
    conv_w = np.random.choice([-1.0, 1.0], size=n_conv).astype(np.float32)
    bn     = np.random.uniform(-2.0, 2.0, size=n_bn).astype(np.float32)
    for x in conv_w: fh.write(f"{x}\n")
    for x in bn:     fh.write(f"{x}\n")
    return n_conv + n_bn

def write_output_conv(fh, inC=64, outC=64, kH=1, kW=1) -> int:
    # Special: no BN, just conv weights in {-1,+1}
    n_conv = conv_elems(inC, outC, kH, kW)
    conv_w = np.random.choice([-1.0, 1.0], size=n_conv).astype(np.float32)
    for x in conv_w: fh.write(f"{x}\n")
    return n_conv

# ----------------------------
# Parse layer_config.h
# ----------------------------
def parse_layer_config(filename="layer_config.h") -> Set[str]:
    """Parse layer_config.h to find which layers are ternary"""
    ternary_layers = set()
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # Look for #define statements that set layers to ternary (1)
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('#define') and 'TERNARY' in line and line.endswith('1'):
                # Extract layer name from #define LAYER_NAME_TERNARY 1
                parts = line.split()
                if len(parts) >= 3 and parts[2] == '1':
                    # Convert LAYER_NAME_TERNARY to layer_name
                    layer_def = parts[1]
                    if layer_def.endswith('_TERNARY'):
                        layer_name = layer_def[:-8].lower()  # Remove _TERNARY suffix
                        ternary_layers.add(layer_name)
        
        print(f"Read configuration from {filename}")
        print(f"Ternary layers: {ternary_layers}")
        return ternary_layers
        
    except FileNotFoundError:
        print(f"{filename} not found, using default configuration")
        return {'input_conv2', 'output_conv'}  # Default ternary layers

# ----------------------------
# UNet layer definitions
# ----------------------------
def write_unet_weights(ternary_layers: Set[str]) -> Tuple[str, int]:
    csv_path = "unet_test_weights.csv"
    total_floats = 0
    
    with open(csv_path, "w") as fh:
        # Input Conv1
        total_floats += write_inconv1(fh, 3, 64, 3, 3)
        
        # Input Conv2 
        if 'input_conv2' in ternary_layers:
            total_floats += write_ternary_layer(fh, 64, 64, 3, 3)
        else:
            total_floats += write_binary_layer(fh, 64, 64, 3, 3)
        
        # Encoder (Down path)
        # DOWN1
        if 'down1_conv1' in ternary_layers:
            total_floats += write_ternary_layer(fh, 64, 128, 3, 3)
        else:
            total_floats += write_binary_layer(fh, 64, 128, 3, 3)
            
        if 'down1_conv2' in ternary_layers:
            total_floats += write_ternary_layer(fh, 128, 128, 3, 3)
        else:
            total_floats += write_binary_layer(fh, 128, 128, 3, 3)
        
        # DOWN2
        if 'down2_conv1' in ternary_layers:
            total_floats += write_ternary_layer(fh, 128, 256, 3, 3)
        else:
            total_floats += write_binary_layer(fh, 128, 256, 3, 3)
            
        if 'down2_conv2' in ternary_layers:
            total_floats += write_ternary_layer(fh, 256, 256, 3, 3)
        else:
            total_floats += write_binary_layer(fh, 256, 256, 3, 3)
        
        # DOWN3
        if 'down3_conv1' in ternary_layers:
            total_floats += write_ternary_layer(fh, 256, 512, 3, 3)
        else:
            total_floats += write_binary_layer(fh, 256, 512, 3, 3)
            
        if 'down3_conv2' in ternary_layers:
            total_floats += write_ternary_layer(fh, 512, 512, 3, 3)
        else:
            total_floats += write_binary_layer(fh, 512, 512, 3, 3)
        
        # DOWN4 (Bottleneck)
        if 'down4_conv1' in ternary_layers:
            total_floats += write_ternary_layer(fh, 512, 1024, 3, 3)
        else:
            total_floats += write_binary_layer(fh, 512, 1024, 3, 3)
            
        if 'down4_conv2' in ternary_layers:
            total_floats += write_ternary_layer(fh, 1024, 1024, 3, 3)
        else:
            total_floats += write_binary_layer(fh, 1024, 1024, 3, 3)
        
        # Decoder (Up path)
        # UP4
        if 'up4_transpose' in ternary_layers:
            total_floats += write_ternary_layer(fh, 1024, 512, 2, 2)
        else:
            total_floats += write_binary_layer(fh, 1024, 512, 2, 2)
            
        if 'up4_conv1' in ternary_layers:
            total_floats += write_ternary_layer(fh, 1024, 512, 3, 3)  # 1024 = 512 + 512 (skip connection)
        else:
            total_floats += write_binary_layer(fh, 1024, 512, 3, 3)
            
        if 'up4_conv2' in ternary_layers:
            total_floats += write_ternary_layer(fh, 512, 512, 3, 3)
        else:
            total_floats += write_binary_layer(fh, 512, 512, 3, 3)
        
        # UP3
        if 'up3_transpose' in ternary_layers:
            total_floats += write_ternary_layer(fh, 512, 256, 2, 2)
        else:
            total_floats += write_binary_layer(fh, 512, 256, 2, 2)
            
        if 'up3_conv1' in ternary_layers:
            total_floats += write_ternary_layer(fh, 512, 256, 3, 3)  # 512 = 256 + 256 (skip connection)
        else:
            total_floats += write_binary_layer(fh, 512, 256, 3, 3)
            
        if 'up3_conv2' in ternary_layers:
            total_floats += write_ternary_layer(fh, 256, 256, 3, 3)
        else:
            total_floats += write_binary_layer(fh, 256, 256, 3, 3)
        
        # UP2
        if 'up2_transpose' in ternary_layers:
            total_floats += write_ternary_layer(fh, 256, 128, 2, 2)
        else:
            total_floats += write_binary_layer(fh, 256, 128, 2, 2)
            
        if 'up2_conv1' in ternary_layers:
            total_floats += write_ternary_layer(fh, 256, 128, 3, 3)  # 256 = 128 + 128 (skip connection)
        else:
            total_floats += write_binary_layer(fh, 256, 128, 3, 3)
            
        if 'up2_conv2' in ternary_layers:
            total_floats += write_ternary_layer(fh, 128, 128, 3, 3)
        else:
            total_floats += write_binary_layer(fh, 128, 128, 3, 3)
        
        # UP1
        if 'up1_transpose' in ternary_layers:
            total_floats += write_ternary_layer(fh, 128, 64, 2, 2)
        else:
            total_floats += write_binary_layer(fh, 128, 64, 2, 2)
            
        if 'up1_conv1' in ternary_layers:
            total_floats += write_ternary_layer(fh, 128, 64, 3, 3)  # 128 = 64 + 64 (skip connection)
        else:
            total_floats += write_binary_layer(fh, 128, 64, 3, 3)
            
        if 'up1_conv2' in ternary_layers:
            total_floats += write_ternary_layer(fh, 64, 64, 3, 3)
        else:
            total_floats += write_binary_layer(fh, 64, 64, 3, 3)
        
        # Output Conv (special handling)
        if 'output_conv' in ternary_layers:
            total_floats += write_ternary_layer(fh, 64, 64, 1, 1)
        else:
            total_floats += write_output_conv(fh, 64, 64, 1, 1)  # No BN for output
    
    print(f"Created {csv_path} with {total_floats:,} floats")
    return csv_path, total_floats

# ----------------------------
# Images
# ----------------------------
def create_test_images(batch_size=1, height=256, width=256, channels=3, path=None) -> str:
    images = np.random.uniform(0.0, 1.0, (batch_size, height, width, channels)).astype(np.float32)
    labels = np.random.randint(0, 2, batch_size, dtype=np.uint8)
    filename = path or f"test_images_{width}x{height}.bin"
    with open(filename, "wb") as f:
        for i in range(batch_size):
            f.write(struct.pack('B', int(labels[i])))
            f.write(images[i].tobytes())
    print(f"Created {filename}: {batch_size}x{width}x{height}x{channels}, "
          f"range [{images.min():.3f},{images.max():.3f}]")
    return filename

if __name__ == "__main__":
    print("=== Creating UNet test weights and images ===")
    
    # Parse configuration
    ternary_layers = parse_layer_config()
    
    # Create weights
    csv_path, total = write_unet_weights(ternary_layers)
    print(f"Total floats written: {total:,}")

    img_path = create_test_images(batch_size=128, height=512, width=512, channels=3)  # Reduced for WMMA FP16 memory requirements
    print("\nFiles created:")
    print(f"  - {csv_path}")
    print(f"  - {img_path}")