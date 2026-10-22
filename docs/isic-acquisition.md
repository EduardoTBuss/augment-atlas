# Aquisição ISIC 2020: operação no x299

Status: **ferramenta pronta e testada offline** (servidor HTTP local, ZIP64 e CSVs
sintéticos). Nada foi baixado e nenhum servidor foi acessado durante o desenvolvimento.
Especificação: [dataset-acquisition-plan.md](dataset-acquisition-plan.md) §3.1.2, §3.3,
§3.4, §3.5, §4, §5, ADR-A8. Este documento diz **como rodar** e onde a implementação se
afasta do plano (com o motivo).

## 1. Desenho: um container com rede para o download, verify sem rede

| Etapa | Onde roda | Rede | Pillow | O que faz |
|---|---|---|---|---|
| `probe` | container pinado | sim | não usa | baixa e confere os 3 CSVs (tamanho, ETag/MD5, cabeçalho exato), faz HEAD do ZIP, lê o EOCD/ZIP64 e guarda o central directory. Não lê o corpo do ZIP |
| `canary --limit 500` | container pinado | sim | sim | stream das primeiras 500 imagens rotuladas + derivado + projeção de tamanho e tempo |
| `full` | container pinado | sim | sim | retoma do ledger e vai até o fim; no final roda `verify` |
| `verify` | container pinado | **`--network none`** | só versões | offline: rehash de todos os derivados, reconciliação, `manifest/*`, `READY.json`, splits D/E e sumário público |

**Por que não separar download (host, stdlib) e derivado (container `--network none`).** O
ZIP cru tem 24,7 GB e a folga útil de `/home` é de cerca de 16 GiB acima da reserva, então
os originais não podem ficar todos no disco. Com spool de 2 GiB, as duas etapas teriam de
se alternar umas 13 vezes (ou mais), ou então rodar ao mesmo tempo como
produtor-consumidor entre dois processos, com dois escritores no ledger. As duas
alternativas trazem mais estado e mais modos de falha do que o desenho atual, que roda
num processo só: o processo principal faz o stream e é o único escritor do ledger, e 2
workers derivam. O que o desenho atual não isola é a decodificação de JPEG não confiável
num container com rede. Isso é mitigado com non-root, `--read-only`, `--cap-drop ALL`,
`no-new-privileges`, `--memory 4g`, `--pids-limit` e um único mount gravável, e o
`verify`, que grava manifesto, READY e sumário, roda sem rede. O `probe` usa só a stdlib e
também roda no host (Python ≥ 3.10) se for preciso. Rodar tudo na mesma imagem evita
depender da versão do Python do host.

Concorrência: um lock exclusivo por raiz (`acquisition-v1/lock`, flock). O processo
principal é o único escritor do ledger SQLite (WAL, `synchronous=FULL`, uma transação por
transição), das renomeações dos derivados e das remoções do spool. Os 2 workers só leem um
arquivo do spool e escrevem um `<imagem>.jpg.tmp` exclusivo. Há no máximo 4 jobs em voo:
os outros membros `FETCHED` esperam num backlog, e a guarda de disco é checada na hora em
que cada job começa de fato. Como só existe um lock, não há ordem de aquisição a
respeitar, e não há deadlock possível.

Ordem de durabilidade por membro: spool `fsync` + rename → ledger `FETCHED` → derivado
`.tmp` `fsync` → rename → ledger `DERIVED` → remoção do spool. Um kill entre quaisquer dois
passos é reparado na sessão seguinte sem reler membros já `DERIVED`/`SKIPPED`: o stream
reabre em `Range` no local header do primeiro membro `PENDING`, com `If-Range` e exigência
de 206.

## 2. Pré-requisitos (uma vez)

```bash
# no x299, dentro de tmux (a sessão sobrevive à queda do SSH)
tmux new -s isic          # ou: tmux attach -t isic

CODE=~/projetos/augment-atlas                  # checkout com a ferramenta, no commit a usar
DATA=~/projetos/augment-atlas/data/isic2020    # raiz privada do dataset (em /home)
IMAGE=sha256:75f24f0e788291fd4005054774994b3945eac91573a8f01834cf35d5e30c434a  # imagem pinada (acceptance-v2)
COMMIT=$(git -C "$CODE" rev-parse HEAD)

git -C "$CODE" status --short                  # precisa estar limpo no commit da ferramenta
docker image inspect --format '{{.Id}}' "$IMAGE"   # precisa devolver exatamente $IMAGE
mkdir -p "$DATA/logs" && chmod 700 "$DATA"
df -B1 --output=avail "$DATA"                  # o canary/full exige >= 33822867456 B (31,5 GiB)
```

`data/` está no `.gitignore` do repositório, então os dados privados não entram no Git.
**Não rode `git clean -x` em `$CODE`**, porque isso apagaria `$DATA`. Se o mesmo checkout
for usado por workers de GPU, prefira mover `DATA` para fora dele (por exemplo
`~/augment-atlas-data/isic2020`). Os comandos abaixo não mudam.

Função auxiliar (cole no shell da sessão tmux):

```bash
acq() {  # uso: acq <bridge|none> <probe|canary|full|verify> [--limit N] [--lenient]
  local net="$1"; shift
  docker run --rm --name isic-acq --network "$net" \
    --user "$(id -u):$(id -g)" --cpus 2 --memory 4g --pids-limit 256 --shm-size 64m \
    --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges \
    -e PYTHONPATH=/code/src -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e HOME=/tmp \
    -e OMP_NUM_THREADS=1 -e AUGMENT_ATLAS_TOOL_COMMIT="$COMMIT" -e AUGMENT_ATLAS_IMAGE_DIGEST="$IMAGE" \
    --mount type=bind,src="$CODE",dst=/code,readonly \
    --mount type=bind,src="$DATA",dst=/data \
    --workdir /tmp --entrypoint python "$IMAGE" \
    -m augment_atlas.acquisition "$@" \
    --config /code/configs/acquisition/isic2020-v1.json --root /data \
    2>&1 | tee -a "$DATA/logs/acq-$(date -u +%Y%m%dT%H%M%SZ)-$1.log"
  local rc=${PIPESTATUS[0]}; echo "exit=$rc"; return "$rc"
}
```

O `--workdir /tmp` (tmpfs vazio) evita que um arquivo do diretório corrente seja importado.
O log fica em `$DATA/logs/` porque pode conter nomes de membro em mensagens de erro, e por
isso é **privado**. O stdout da CLI (uma linha JSON) traz só contagens e hashes.

## 3. Sequência exata

```bash
# 1) probe: CSVs + HEAD + central directory (minutos; ~4,5 MB baixados + o CD do ZIP)
acq bridge probe
#    exit 0 => sem divergência pendente. exit 3 => ler $DATA/manifest/divergences.json (seção 5).

# 2) canary: 500 imagens rotuladas (cerca de 375 MB de stream)
acq bridge canary --limit 500
python3 - "$DATA/acquisition-v1/canary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
p, t = d["projection"], d["timing"]
print("derivados", d["derived"], "| projecao GiB %.3f (meta <= 6, teto 8)" % (p["projected_bytes"] / 2**30))
print("s/imagem %.3f com %d workers -> estimativa h %.2f" % (t["seconds_per_derived_image"], t["workers"],
      t["seconds_per_derived_image"] * p["remaining"] / 3600))
print("decoder", d["decoder"], "| min headroom GiB %.2f" % (d["disk_min_headroom_bytes"] / 2**30))
PY

# 3) completo: retoma depois do canary (não relê o que já foi derivado); no fim roda o verify
acq bridge full

# 4) verify sem rede (idempotente; pode repetir a qualquer momento depois do full)
acq none verify
```

Retomada: **repita o mesmo comando** (`acq bridge full`) depois de qualquer exit 75. Parar
de propósito: `docker stop -t 60 isic-acq`. O SIGTERM vira pausa limpa (exit 75), com a
sessão registrada no ledger.

Pontos de controle do canary (F3 do plano): projeção ≤ 6 GiB (acima de 8 GiB a ferramenta
para com exit 3), tempo por imagem e `decoder` (Pillow e libjpeg-turbo da imagem pinada).
Reprodutibilidade opcional do derivado entre duas execuções (a segunda baixa só cerca de
40 MB). Faça isso logo depois do canary e antes do `full`:

```bash
MAIN="$DATA"; DATA=~/projetos/augment-atlas/data/isic2020-repro; mkdir -p "$DATA/logs" && chmod 700 "$DATA"
acq bridge canary --limit 50
python3 - "$MAIN/acquisition-v1/ledger.sqlite" "$DATA/acquisition-v1/ledger.sqlite" <<'PY'
import sqlite3, sys
q = "SELECT idx, info FROM member WHERE state='DERIVED'"
main, repro = (dict(sqlite3.connect(p).execute(q).fetchall()) for p in sys.argv[1:])
same = len(repro) >= 45 and all(main.get(k) == v for k, v in repro.items())
print("identicos (%d)" % len(repro) if same else "DIFERENTES: nao seguir")
PY
DATA="$MAIN"   # depois, apagar a raiz -repro manualmente
```

## 4. Códigos de saída

| Exit | Significado | Ação |
|---|---|---|
| 0 | `probe` limpo, `canary` dentro do orçamento, ou `READY.json` escrito | seguir |
| 75 | pausa retomável: guarda de disco, rede (8 tentativas com backoff esgotadas), lock ocupado, SIGTERM, `verify` antes do fim | resolver a causa (espaço, rede) e repetir o mesmo comando |
| 3 | divergência que precisa de decisão humana: cabeçalho de CSV diferente (ADR-A8, sem mapeamento automático), ETag/tamanho mudou, contagem diferente da documentada, projeção > 8 GiB, auditoria de split reprovada (sem retry de seed), identidade do config mudou depois do início do stream | ler `manifest/divergences.json` e decidir (seção 5) |
| 2 | falha de integridade (CRC32/tamanho vs CD, ZIP hostil, hash de derivado diferente do ledger, lint de privacidade) | não repetir; investigar |
| 64 | uso ou config inválido | corrigir o comando ou o config |

## 5. Divergências

Cada divergência é registrada em `manifest/divergences.json` com `documented`, `observed`
e até 10 exemplos (privados). No modo padrão (estrito), `canary`/`full` não fazem stream
com divergência pendente. `--lenient` faz o stream mesmo assim, mas o READY continua
bloqueado. Para aceitar uma divergência, acrescente ao config, em `accepted_divergences`
(chave operacional, que não muda a identidade fixada no ledger):

```json
{"field": "counts.melanoma_images", "observed": 581, "resolution": "<motivo>", "decided_by": "Eduardo"}
```

A aceitação só vale para o mesmo `field` **e** o mesmo `observed`. Depois de aceitar,
rode `acq none verify`. Quem escreve a decisão é o Eduardo. O config versionado é público
e não leva IDs (a divergência só tem contagens).

Checagens que geram divergência: contagens documentadas (33.126 imagens, 2.056 pacientes,
584 melanomas, 425 pares e 850 nomes de duplicata); bijeção entre os membros `.jpg` do ZIP e
o `image_name` do GroundTruth v2; v1 × v2 (`image_name`, `patient_id` e `target` 1:1);
`target==1 ⇔ diagnosis=="melanoma" ⇔ benign_malignant=="malignant"`; formato da lista de
duplicatas (pares repetidos, auto-pares, pares não disjuntos, nomes fora do GroundTruth);
par oficial sem SHA256 idêntico do original, e vice-versa; imagens recusadas pelo
decodificador. São falhas duras (exit 3 imediato): `patient_id` vazio (nunca imputado),
`target` fora de {0,1}, `image_name` repetido ou fora do padrão.

## 6. Disco

- Pré-checagem (passo 1 do plano), consciente da retomada: `livre − 20 GiB ≥ (8 GiB −
  já derivado) + (2 GiB − spool atual) + 1,5 GiB`. Numa raiz nova isso exige 31,5 GiB
  livres; com cerca de 39 GB (36,3 GiB) em `/home` sobram cerca de 4,8 GiB de folga.
- Antes de cada escrita de `s` bytes (membro no spool, derivado): `livre − s − 20 GiB ≥
  1 GiB`, medido no filesystem real. Por isso a guarda também cobre checkpoints de jobs
  de GPU. Se a condição falhar, a sessão para limpa, sem `.tmp`, com exit 75.
- O ZIP nunca é gravado inteiro. O spool guarda no máximo 2 GiB de originais. Ao atingir
  o limite, a conexão é fechada, o spool drena até menos de 1 GiB e o stream reabre no
  membro seguinte. Os originais derivados são apagados (o único arquivo que a ferramenta
  apaga é o próprio spool e os `.tmp`/`.part` dela).

## 7. Saídas (todas privadas, sob `$DATA`)

```
acquisition-v1/{config.json, probe.json, canary.json, central-directory.bin, ledger.sqlite, lock, spool/}
release/ISIC_2020_Training_{GroundTruth,GroundTruth_v2,Duplicates}.csv   # retidos byte a byte
derived-512-v1/images/<image_name>.jpg                                   # isic2020-derived-512-v1
manifest/{manifest.csv, records.json, manifest-private.json, divergences.json, READY.json}
splits/global-dev-eval-v1/{pools.json, folds.json, audit.json}
public/isic2020-summary-public.json                                      # agregado, sem IDs
logs/                                                                    # logs do tee (privados)
```

`manifest.csv` segue o contrato `manifest_csv` (`path,group,label,sha256`, com caminhos
relativos a `$DATA`). Para treinar: `dataset.root=$DATA`,
`dataset.manifest=manifest/manifest.csv`, `dataset.manifest_sha256=<READY.files["manifest.csv"]>`
e `require_image_hashes: true`. Só o `public/isic2020-summary-public.json` sai do servidor.
Antes de ser escrito, ele passa por um lint de regex (`ISIC_\d{7}`, `IP_\d{7}`, `IL_\d{7}`)
e de todos os IDs conhecidos da execução (imagens, pacientes, lesões, grupos, nomes de
membro). Qualquer ocorrência resulta em exit 2 e o arquivo não é escrito.

Splits: `global-dev-eval-v1` com `split_seed=20261019` (fixado no config antes do
download), D=0,20, `f_sel`=0,30, 5 folds em E e limiares pré-registrados (D_select ≥ 15
grupos positivos, cada E_f ≥ 50). Se a auditoria reprovar: exit 3, sem `pools.json`, e
sem nova seed.

**Emenda de 22/10/2026 (decisões S1/S2 do Eduardo).** Os pools usados pelo protocolo não
saem mais deste split de seed única. Eles vêm de `global_cli make-pools` com a emenda
pré-registrada `configs/acquisition/isic2020-v1-splits-amendment-20261022.json`, que fixa:
- seeds explícitas: development = selection = 20261019 e evaluation = 20261022;
- quartis nearest-rank;
- arredondamento meio-para-cima exato;
- um contador único de folds.

A saída `splits/global-dev-eval-v1/` do `verify` continua sendo escrita, mas é um
diagnóstico pré-emenda: não é importada. Um exit 3 do `verify` causado **só** pela
auditoria desse split antigo, com o `READY.json` já escrito, não bloqueia nada
(`docs/x299-runbook.md` §B1). O bloco `splits` do config não foi editado, porque faz
parte da identidade gravada no ledger do download.

**Oito pares extras (22/10/2026).** Na verificação apareceram 8 pares de originais
byte-idênticos que não estão na lista oficial. Eles foram aceitos em
`accepted_divergences` (`duplicates.identical_originals_not_listed`, `observed: 8`), que é
uma chave operacional, então a identidade não muda. O `verify` une esses pares no
union-find e aplica `keep-min-image-name-v1`. Não há novo download.

## 8. Diferenças em relação ao plano (deliberadas)

| Plano | Implementado | Motivo |
|---|---|---|
| `isic2020 {probe\|run\|verify\|summarize}` | `probe \| canary --limit N \| full \| verify`; o sumário é escrito pelo `verify` | pedido do maestro (canary explícito); o sumário depende do READY |
| `python -I -m ...` | `PYTHONPATH=/code/src` com `--workdir /tmp` | `-I` ignora `PYTHONPATH`; o tmpfs vazio cumpre o mesmo papel |
| `~/augment-atlas-private/datasets/isic2020/` | `~/projetos/augment-atlas/data/isic2020/` | pasta definida pelo Eduardo (`data/` está no `.gitignore`) |
| Retomada por chunk de 32 MiB + `zip_chunked_root` | retomada por membro (Range no local header do primeiro `PENDING`); `chunked_sha256_root: null` | a identidade fica com ETag + tamanho + SHA256 do central directory + CRC32 e SHA256 por membro; um membro interrompido é relido inteiro (≤ alguns MB) |
| `sha256` do ZIP inteiro em sessão única | não calculado (`null` com motivo) | o S3 só dá ETag multipart; o CD e os membros são hasheados |
| quase-duplicatas por dHash (só relatório) | não implementado (`near_duplicate_pairs: null`) | não bloqueia nada; fica como dívida |
| `config` `isic2020-derived-512-v1.json` | `configs/acquisition/isic2020-v1.json` | nome pedido pelo maestro |
| duplicatas "425 imagens" | 425 **pares**, 850 nomes (§3.1.2) | fato verificado |

## 9. Testes

`PYTHONPATH=src python -m unittest discover -s tests -v`. São específicos da parte B:
`tests/test_acquisition_isic.py` (unidades e política de IDs do repositório) e
`tests/test_acquisition_isic_e2e.py` (servidor local 127.0.0.1, ZIP64 sintético com
diretório, lixo `__MACOSX`/`.DS_Store`, stored + deflate + data descriptor). Cobrem: probe
→ full → READY, com o `manifest.csv` carregado pelo `data.load_records`; canary limitado
seguido de full que retoma no primeiro `PENDING`; queda de conexão no meio de um membro;
backpressure com 2 workers e spool mínimo; kill abrupto (`os._exit`) durante a derivação
e na guarda de disco, seguido de retomada; SIGTERM vira pausa limpa; disco insuficiente na
pré-checagem (nada é escrito) e no meio da execução (exit 75 sem `.tmp`, depois retomada);
cabeçalho divergente → exit 3; ETag trocado → exit 3; identidade do config trocada → exit
3; contagens divergentes registradas, estrito × `--lenient`, aceitação → READY; sumário
público varrido contra todos os IDs; derivado real com Pillow (pool de 2 processos). Os
fixtures usam só nomes sintéticos (`SYNTH_0000001`, `PSYNTH_0001`).
