#!/usr/bin/env python3
"""
Interactive script to create cone map from manual measurements
or from rosbag playback
"""

import json
import sys

def create_cone_map_manually():
    """Create cone map by entering positions manually"""
    print("=================================")
    print("Manual Cone Map Creator")
    print("=================================")
    print("Enter cone positions. Type 'done' when finished.\n")
    
    cones = []
    cone_id = 0
    
    while True:
        print(f"\nCone #{cone_id}")
        label = input(f"  Label (default: Cone_{cone_id}): ").strip()
        if label.lower() == 'done':
            break
        if not label:
            label = f"Cone_{cone_id}"
        
        try:
            x = float(input("  X coordinate (m): "))
            y = float(input("  Y coordinate (m): "))
            z = float(input("  Z coordinate (m, default 0.0): ") or "0.0")
        except ValueError:
            print("Invalid input. Please enter numbers.")
            continue
        
        cones.append({
            'id': cone_id,
            'label': label,
            'x': x,
            'y': y,
            'z': z
        })
        
        cone_id += 1
        
        cont = input("\nAdd another cone? (y/n): ").lower()
        if cont != 'y':
            break
    
    # Save to file
    filename = input("\nSave to file (default: cone_map.json): ").strip()
    if not filename:
        filename = "cone_map.json"
    
    with open(filename, 'w') as f:
        json.dump(cones, f, indent=2)
    
    print(f"\nCone map saved to {filename}")
    print(f"Total cones: {len(cones)}")
    for cone in cones:
        print(f"  {cone['label']}: ({cone['x']:.2f}, {cone['y']:.2f}, {cone['z']:.2f})")

if __name__ == '__main__':
    create_cone_map_manually()
