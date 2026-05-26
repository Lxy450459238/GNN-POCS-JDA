import subprocess, sys, time
# Use ssh.exe directly (Windows OpenSSH)
key_file = r"C:\Users\John\.ssh\id_rsa.pub"
with open(key_file) as f:
    key = f.read().strip()

# Write key to temp file first
with open(r"C:\Users\John\.ssh\key_to_copy.txt", "w") as f:
    f.write(key)

print("Key prepared. Run this manually in a new terminal:")
print(f'ssh ubuntu@10.246.4.76 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys" < C:\\Users\\John\\.ssh\\key_to_copy.txt')
print()
print("Password: 123.com")
