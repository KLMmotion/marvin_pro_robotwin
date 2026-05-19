#!/usr/bin/env python3
import os
import sys
import glob
import yaml

def print_color(message, color_code):
    NC = '\033[0m'
    print(f"{color_code}{message}{NC}")

BLUE = '\033[0;34m'
YELLOW = '\033[0;33m'
GREEN = '\033[0;32m'
RED = '\033[0;31m'


def prompt_path():
    answer = input("Do you want to manually specify the absolute path to the assets directory? (y/n): ")
    if answer.lower() != 'y':
        sys.exit(1)
    return input("Please enter the absolute path: ")


def validate_generated_config(config_file):
    with open(config_file, 'r') as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)

    kinematics = data.get('robot_cfg', {}).get('kinematics', {})
    missing_paths = []
    for key in ('urdf_path', 'collision_spheres'):
        path = kinematics.get(key)
        if path and not os.path.exists(path):
            missing_paths.append((key, path))

    return missing_paths


def find_template_files(repo_root):
    embodiment_config_path = os.path.join(repo_root, 'task_config', '_embodiment_config.yml')
    if not os.path.exists(embodiment_config_path):
        pattern = os.path.join(repo_root, 'assets', 'embodiments', '**', '*_tmp.yml')
        return glob.glob(pattern, recursive=True)

    with open(embodiment_config_path, 'r') as f:
        embodiment_config = yaml.load(f.read(), Loader=yaml.FullLoader) or {}

    config_files = []
    seen = set()
    for embodiment_name, embodiment_info in embodiment_config.items():
        embodiment_path = embodiment_info.get('file_path')
        if not embodiment_path:
            print_color(f"  ! Warning: {embodiment_name} has no file_path", YELLOW)
            continue
        if not os.path.isabs(embodiment_path):
            embodiment_path = os.path.join(repo_root, embodiment_path)
        pattern = os.path.join(embodiment_path, '**', '*_tmp.yml')
        for config_file in glob.glob(pattern, recursive=True):
            if config_file not in seen:
                seen.add(config_file)
                config_files.append(config_file)

    return config_files


def main():
    # Get current directory
    assets_path = os.getcwd()
    print_color(f"Current path: {assets_path}", BLUE)

    # Check ./assets/embodiments
    if not os.path.isdir(os.path.join(assets_path, 'assets', 'embodiments')):
        print_color("Warning: ./assets/embodiments directory not found", YELLOW)
        parent = os.path.abspath(os.path.join(assets_path, '..'))
        if os.path.isdir(os.path.join(parent, 'assets', 'embodiments')):
            print("Found assets/embodiments in parent directory, switching...")
            assets_path = parent
            os.chdir(assets_path)
            print_color(f"Updated path: {assets_path}", BLUE)
        else:
            print_color("Please ensure you're running this script in the correct directory", YELLOW)
            print("Script should be run in the repository root directory containing assets/embodiments")
            assets_path = prompt_path()
            if not os.path.isdir(os.path.join(assets_path, 'assets', 'embodiments')):
                print_color("Error: Cannot find assets/embodiments directory at the specified path", YELLOW)
                sys.exit(1)
            os.chdir(assets_path)
            print_color(f"Switched to: {assets_path}", BLUE)

    # Export environment variable
    os.environ['ASSETS_PATH'] = assets_path
    print_color(f"Setting environment variable: ASSETS_PATH={assets_path}", BLUE)

    # Counters
    count_total = count_updated = count_error = 0

    # Find *_tmp.yml files
    print_color("Searching for configuration template files...", BLUE)
    config_files = find_template_files(assets_path)

    if not config_files:
        print_color("No *_tmp.yml files found", YELLOW)
        sys.exit(1)

    print_color("Starting to process configuration files...", BLUE)
    for tmp_file in config_files:
        count_total += 1
        target_file = tmp_file.replace('_tmp.yml', '.yml')
        print(f"Processing [{count_total}]: {tmp_file} -> {target_file}")

        try:
            with open(tmp_file, 'r') as f:
                content = f.read()

            has_placeholder = ('${ASSETS_PATH}' in content) or ('$ASSETS_PATH' in content)
            if not has_placeholder:
                print_color("  ! Warning: template has no ASSETS_PATH placeholder", YELLOW)

            new_content = content.replace('${ASSETS_PATH}', assets_path)
            new_content = new_content.replace('$ASSETS_PATH', assets_path)

            with open(target_file, 'w') as f:
                f.write(new_content)

            print_color(f"  ✓ Successfully replaced ${{ASSETS_PATH}} -> {assets_path}", GREEN)
            count_updated += 1

            if has_placeholder and assets_path in new_content:
                print_color("  ✓ Confirmed path was correctly replaced", GREEN)
            elif has_placeholder:
                print_color("  ! Warning: Could not confirm if path was correctly replaced", YELLOW)

            missing_paths = validate_generated_config(target_file)
            if missing_paths:
                for key, path in missing_paths:
                    print_color(f"  ✗ Missing {key}: {path}", RED)
                count_error += 1
            else:
                print_color("  ✓ Confirmed referenced files exist", GREEN)

        except Exception as e:
            print_color(f"  ✗ Replacement failed: {e}", YELLOW)
            count_error += 1

    # Summary
    print()
    print_color("Processing complete!", BLUE)
    print(f"Total processed: {count_total} files")
    print_color(f"Successfully updated: {count_updated} files", GREEN)
    if count_error > 0:
        print_color(f"Failed to process: {count_error} files", YELLOW)
        sys.exit(1)

    print()
    print_color("All template files have been processed!", GREEN)
    print("To use in a new environment, run this script again")


if __name__ == '__main__':
    main()
