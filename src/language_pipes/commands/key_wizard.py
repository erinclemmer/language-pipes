import os

from language_pipes.util.aes import save_new_aes_key
from language_pipes.util.user_prompts import prompt_bool, prompt

def key_wizard(default_path):
    print("--- Step 1: Network Key ---\n")
    print("A network key is required for secure communication between nodes.")
    print("All nodes in your network must share the same key.\n")
    
    use_existing = prompt_bool("Use an existing key file?", default=False)
    if use_existing:
        selected_path = prompt("Existing key file", default=default_path, required=True)
        if os.path.exists(selected_path):
            print(f"✓ Network key found at '{selected_path}'")
            return selected_path
        else:
            while True:
                print(f"Error cannot find file at {selected_path}")
                again = prompt_bool("Try again?", default=True)
                if not again:
                    use_existing = False
                    break
                selected_path = prompt("Existing key file", default=default_path, required=True)

    if not use_existing:
        is_first = prompt_bool("Are you the first node in this network?", default=True)
        if is_first:
            create_key = prompt_bool("Use encrypted connection for network data?", default=False)
            if create_key:
                selected_path = prompt("Path for key generation", default=default_path, required=True)
                print(f"\nGenerating new network key...")
                key = save_new_aes_key(selected_path)
                print(f"✓ Network key generated: {key}")
                print(f"✓ Network key saved to '{selected_path}'")
                print("\n  IMPORTANT: Copy this key file to all other nodes that will")
                print("  join your network. Keep it secure - anyone with this key can join.\n")
                input("Press Enter to continue")
                return selected_path
            else:
                print("A network key is not needed, continuing...")
                return None
        else:
            use_key = prompt_bool("Does the network use encryption?", default=False)
            if use_key:
                print("\n  Enter the network key from your first node.")
                print("  (This is the hex string that was displayed when the key was generated)\n")
                key_value = prompt("Network key: ", required=True)
                
                with open(default_path, 'w', encoding='utf-8') as f:
                    f.write(key_value)
                
                print(f"✓ Network key saved to '{default_path}'")
                return default_path
            else:
                print("Continuing initialization...")
                return None
    return None
