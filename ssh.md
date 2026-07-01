# RunPod SSH Public Key — Arc2Avatar

**Public key for RunPod GPU instance access:**

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIBAfPpBAmD/W8Y+duizVYHQqbIjGFoMIY5lYTVEyZrc runpod-arc2avatar
```

## Usage

1. **Add to RunPod** — paste the key above into your RunPod account's SSH Keys section (Settings → SSH Keys)
2. **Connect**:
   ```bash
   ssh -i runpod_ssh_key root@<runpod-instance-ip> -p <port>
   ```
3. **Private key location**: `runpod_ssh_key` (same directory, keep secure, never share)
4. **Permissions**: `chmod 600 runpod_ssh_key` if needed

## Key Details
- **Type**: Ed25519 (256-bit)
- **Created**: 2026-07-01
- **Purpose**: GPU instance access for Arc2Avatar pipeline migration
