# Fixtures reais do nó x299 (sanitizadas)

- Capturadas em 2026-10-19T12:38:41Z por Claude (maestro-software) via SSH, somente leitura, servidor ocioso (sem containers ativos).
- Script: comandos da seção 8.1 de `docs/fleet-protocol.md` (sensors -j/texto, nvidia-smi query/stream 10 s/-q, powercap, hwmon, thermal, /proc, /var/crash, timedatectl, lscpu).
- Driver NVIDIA 580.178.04; kernel 6.8.0-142-generic; 2× GTX TITAN X 12 GB.
- Sanitização: UUIDs de GPU → `GPU-00000000-0000-0000-0000-00000000000{0,1}`; nome de usuário → `atlasuser`. Hostname não aparece. boot_id mantido (aleatório por boot, não identifica pessoa).
- `energy_uj.attempt.txt` registra o "Permissão negada" real (RAPL root-only).
- Os bytes brutos não sanitizados ficam fora do Git (pasta privada da sessão).
