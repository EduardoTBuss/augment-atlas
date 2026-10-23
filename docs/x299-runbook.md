# Runbook do servidor (X299) — validação técnica, ISIC offline e piloto INbreast exploratório

Status: **preparado na nuvem, não executado.** Tudo o que este documento descreve foi
exercitado apenas com fixtures sintéticas, na sessão de nuvem, sem servidor, GPU ou dados
reais. Nenhum passo aqui foi validado no X299. Os resultados do servidor são do Eduardo e
ficam fora do repositório público.

- O piloto INbreast é **exploratório**. Não é confirmação científica nem conclusão do
  artigo. `scientific_status` continua `"provisional"`, e `execution_verified` e
  `scientific_eligible` continuam `false` em todos os documentos.
- Escopo desta etapa: **INbreast e ISIC 2020**. mBRSET (ordinal) e CheXpert (multilabel)
  aguardam acesso ao PhysioNet e não têm caminho de código nesta versão.
- Pendentes segundo o Brain, e **não substituídos por este runbook**: o **Gate V4** e o
  **ensaio de endurance de 5h30** (`docs/endurance-test.md`). A Parte C só começa depois
  de os dois estarem registrados como aprovados no Brain.

Referências: [`cloud-context.md`](cloud-context.md) (estado e decisões),
[`global-cli.md`](global-cli.md) (comandos), [`isic-acquisition.md`](isic-acquisition.md)
(ferramenta ISIC), [`global-next-integration.md`](global-next-integration.md) §10 (piloto),
[`endurance-test.md`](endurance-test.md) (guardião e limites), `README.md` (worker, AVX2).

---

## 0. Convenções e placeholders

Nenhum caminho privado, hostname ou credencial entra no Git. Use estes placeholders e
defina-os numa sessão `tmux` privada:

```bash
CODE=<checkout limpo do repositório, no commit pinado>      # nunca com dados dentro
COMMIT=<commit pinado; ver §1>                               # 40 hex
IMAGE=<ID imutável da imagem, sha256:...; ver §1>
INB=<raiz privada de dados do INbreast>                      # settings.data dos workers INbreast
ISIC=<raiz privada do ISIC (a raiz da aquisição já baixada)>
VIEWS=<raiz privada das D-views>                             # settings.views dos workers HPO
RUNS=<raiz privada dos runs>                                 # settings.runs
ADMIN=<checkout de fila só do operador>                      # nunca o de um worker
REMOTE=<URL do remoto da fila>                               # branch queue
PRIV=<diretório privado do operador, modo 700>               # settings, probe, specs, recibos
G="python3 -m augment_atlas.global_cli"
export PYTHONPATH="$CODE/src"
```

Regras de operação válidas em todas as partes:
- **Um comando por vez.** Leia a saída JSON (só hashes, contagens e commits) e anote os
  hashes da §5 antes do próximo passo. Exit 0 significa seguir. Exit 2 significa parar e
  ler o código de erro (prefixo `codigo:`). Nunca repita com outra seed ou outro parâmetro
  para "fazer passar".
- **Os comandos são idempotentes.** Repetir um passo com as mesmas entradas devolve o
  mesmo hash e não sobrescreve nada (no-clobber). Isso é a retomada padrão.
- **Nunca rode `git clean -x`** num checkout que tenha dados por perto, e nunca edite à mão
  arquivos em `<raiz>/global/<dataset>/`: eles são endereçados por hash e reauditados.
- O guardião térmico/recursos privado (V2) continua sendo a proteção independente, e o
  worker mantém o próprio limite `max_temperature_c`. Este runbook não cria outra execução
  nem desliga guarda nenhuma.

## 1. Pin de código e imagem

1. Os PRs #3 e #4 precisam estar mergeados, nesta ordem, num commit que o Eduardo revisou.
   `COMMIT` é esse commit, ou um descendente que mude só `configs/global/**`.
2. `git -C "$CODE" fetch origin && git -C "$CODE" checkout "$COMMIT" && git -C "$CODE" status --porcelain`
   tem de sair vazio. O `make-pools` exige que o config e a emenda em `COMMIT` sejam
   idênticos aos de `origin/main` (por isso o `fetch`).
3. `IMAGE`: a imagem pinada nos aceites anteriores (`docs/acceptance-v2.md`, também em
   `docs/isic-acquisition.md` §2). Confira com `docker image inspect --format '{{.Id}}' "$IMAGE"`,
   que precisa devolver exatamente `$IMAGE`. Uma tag não basta.
4. Pesos ImageNet: `IMAGENET1K_V2`, sha256
   `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca` (o mesmo de
   `configs/smoke/candidate-v2/*.json`). O worker confere o arquivo do `torch_cache` antes de
   qualquer claim.
5. Probe do ambiente (vai para `environment_sha256` do freeze e do plano):

```bash
docker run --rm --network none --gpus device=0 --entrypoint python "$IMAGE" \
  -c "$(python3 -c 'import augment_atlas.runtime as r; print(r.PROBE)')" > "$PRIV/probe.json"
```

## Parte A — Validação técnica do host e do código (sem dados clínicos, sem campanha)

| Passo | Comando | Esperado | Parar se |
|---|---|---|---|
| A1 suíte offline, **usuário sem privilégios** | `PYTHONPATH=src python3 -m unittest discover -s tests` em `$CODE` | `OK`, 0 falhas; testes pulados só os de torch/contêiner (contagens em `cloud-context.md` §4) | qualquer falha |
| A2 suíte como root (opcional) | o mesmo comando como root | só `test_sentinel…test_cli_run_report_latest` falha: o sentinel recusa root por desenho (F1) | qualquer outra falha |
| A3 imagem e pesos | §1 passos 3 e 4 | IDs e hashes idênticos | divergência |
| A4 limite AVX2 | `settings` sem `cpu_isa_max` (padrão `avx2`); depois de um job, `runtime.json` registra `cpu_isa_max` | `avx2` | outro valor sem decisão registrada |
| A5 sentinel só de observação | `./scripts/sentinel-observe-only.sh --duration 1800 --node-id <alias>` (não roda como root) | diretório privado novo; `python3 -m augment_atlas.sentinel report --dir <dir> --validate` sem erros | erro de validação |
| A6 worker smoke sintético | o fluxo de `README.md` "Quick start" (`cli seed --phase smoke` com `configs/smoke/synthetic.json`, depois `./scripts/worker.sh --settings … --phase smoke --once`) | job `succeeded`, artefatos validados | falha, temperatura no limite |

Gate V4 e endurance de 5h30 (§ topo): executar e registrar no Brain conforme o plano
privado deles, **antes** da Parte C. Este repositório não traz o orquestrador nem o
guardião desses ensaios.

## Parte B — ISIC 2020: verificação offline do dataset já baixado e pools D/E

O download já foi feito; **não baixe de novo**. A identidade do config não mudou
(`605233a8…` em `cloud-context.md`), então o ledger existente continua válido.

### B1. `verify` sem rede, com as emendas de 22/10/2026

Use a função `acq` de `docs/isic-acquisition.md` §2. Ela monta `$DATA`, então defina
`DATA="$ISIC"` antes. `CODE` deve estar no `COMMIT` pinado, que contém a entrada
`accepted_divergences` dos oito pares extras byte-idênticos.

```bash
acq none verify          # offline; repete sem custo
echo "exit=$?"
```

| Exit | Leitura | Ação |
|---|---|---|
| 0 | `manifest/READY.json` escrito; os 8 pares extras entram no union-find (`duplicates.exact_hash_pairs` = oficiais idênticos + extras) | seguir para B2 |
| 3 com `split audit failed` e `READY.json` presente | é o split **antigo, de seed única**, do `verify` (diagnóstico pré-emenda, nunca importado) | anotar e seguir para B2: os pools vêm de `make-pools` com a emenda |
| 3 com divergência pendente (`manifest/divergences.json`) | uma divergência que não é a dos 8 pares (outro campo ou outro `observed`) | **parar**; a decisão é do Eduardo (`isic-acquisition.md` §5) |
| 2 | integridade | **parar**; investigar, não repetir |
| 75 | pausa retomável (disco, lock) | resolver e repetir o mesmo comando |

Confira em `$ISIC/manifest/divergences.json` que o item
`duplicates.identical_originals_not_listed` tem `observed: 8` e `accepted: true`. Se o
`observed` for outro número, **pare**: a aceitação vale só para 8.

### B2. Spec, pools D/E (`global-dev-eval-v1`) e selo

```bash
MAN=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["files"]["manifest.csv"])' "$ISIC/manifest/READY.json")
printf '{"adapter":"manifest_csv","manifest":"manifest/manifest.csv","manifest_sha256":"%s"}\n' "$MAN" > "$PRIV/isic-source.json"
I="--data-root $ISIC --source $PRIV/isic-source.json --dataset isic2020"
AMD="$CODE/configs/acquisition/isic2020-v1-splits-amendment-20261022.json"

$G split-spec $I --acquisition-root "$ISIC" --split-amendment "$AMD" --out "$PRIV/isic-split-spec.json"
$G make-pools $I --spec "$PRIV/isic-split-spec.json" --code "$CODE" --acquisition-root "$ISIC" \
   --acquisition-config "$CODE/configs/acquisition/isic2020-v1.json" --split-amendment "$AMD"
```

`make-pools` aceita só a emenda e o config **commitados e sem alteração** no `--code`
limpo, e só uma spec igual à derivada da raiz verificada. Ele imprime:
- `candidate_sha256` e `global_manifest_sha256`;
- as seeds (20261019, 20261019, 20261022);
- os hashes da spec, do config e da emenda;
- as contagens por pool.

- **Se sair com `split_audit_failed`, pare e escale.** Nenhuma seed nova, nenhum fallback
  (decisão S1).
- Os limiares são D_select ≥ 15 grupos positivos e cada E_f ≥ 50.

Com as contagens aprovadas pelo Eduardo:

```bash
$G import-pools $I --file <path impresso> --expected-manifest <global_manifest_sha256>
$G audit-pools  $I --pools-sha256 <pools_sha256> --expected-manifest <global_manifest_sha256>
$G anchor $I --pools-sha256 <pools_sha256> --code "$CODE" --remote "$REMOTE" --checkout "$ADMIN" \
   --kind pools --document-sha256 <pools_sha256> --expected-manifest <global_manifest_sha256> --document-utc <UTC agora>
```

**A Parte B termina aqui.** HPO e treinos finais do ISIC **não** fazem parte desta etapa,
por dois motivos:
- a Q9 manda subamostrar pacientes com N declarado antes do split, e N ainda não foi fixado;
- o orçamento de GPU do ISIC completo não foi calibrado.

O código do caminho HPO serve aos dois datasets binários. Para o ISIC, ele só roda depois
dessas duas decisões.

## Parte C — Piloto INbreast exploratório (duas campanhas, teto de 8 GPU-h)

Pré-requisitos: Parte A aprovada; Gate V4 e endurance de 5h30 aprovados no Brain; guardião
V2 armado; `max_temperature_c` do worker conforme o teto vigente; um treino por vez
(GPU serial).

**Ordem obrigatória (decisão 7):**
1. receita externa: freeze, âncora, configs finais (a), campanha (a) selada e ancorada;
2. só então os jobs HPO;
3. freeze `hpo_d`, âncora, configs finais (b), campanha (b) selada e ancorada;
4. só então qualquer job final.

O código recusa a quebra dessa ordem:
- num dataset exposto, o `seed-hpo` exige a âncora da campanha (a);
- num dataset exposto, o `seed-final` exige as âncoras das duas campanhas;
- a cronologia da campanha (b) recusa qualquer job final dos pools criado antes da âncora
  da campanha (b) (`final_job_before_seal`).

**Teto de 8 GPU-h para o piloto inteiro.** Ao atingir o teto, pare os workers, registre a
matriz incompleta e **não escolha quais jobs completar**. A estimativa da spec é de cerca
de 4–5 GPU-h por campanha de 50 jobs. Com duas campanhas, o teto pode ser atingido antes do
fim, e isso é um resultado esperado e reportável.

```bash
N="--data-root $INB --source $PRIV/inbreast-source.json --dataset inbreast"
Q="--remote $REMOTE --checkout $ADMIN"
RT="--code $CODE --image-id $IMAGE --environment-probe $PRIV/probe.json \
    --weights-sha256 11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
```

`$PRIV/inbreast-source.json`: `{"adapter": "inbreast_csv", "manifest": "<manifest com hash
do endurance, relativo a $INB>", "manifest_sha256": "<sha256 desse arquivo>"}`.

### C0. Registros e pools

1. `$G identity $N --num-classes 2`: espera **387** registros. Anote `records_sha256`.
2. Os pools do INbreast já existem (os privados de 16/10, seeds 17/29/31). **Não gere pools
   novos para o INbreast.** Se o arquivo não estiver no schema v1, pare: a conversão das
   mesmas listas para v1 é uma tarefa de código à parte (sem novo sorteio).
3. `$G import-pools $N --file <pools privados> --expected-manifest <hash pinado no Brain>`.
   O `dataset.id` precisa ser `inbreast` com `prior_exposure: known`. Depois rode
   `audit-pools` e `anchor --kind pools`.
4. Calibração de throughput: 1 job `smoke` com dados de D ou sintéticos, **nunca de E**
   (spec §10, passo 2), pelo fluxo smoke normal. Recalcule a estimativa de GPU-h.

### C1. Campanha (a): receita externa fixa (Q2)

```bash
$G seal-external $N --pools-sha256 $P $RT --final-seeds <s1,s2,s3> --color grayscale_to_rgb --frozen-at <UTC>
$G anchor $N --pools-sha256 $P --code $CODE $Q --kind recipe_freeze --document-sha256 $FA   # C_freeze (a)
$G materialize-final $N --pools-sha256 $P --code $CODE --freeze-sha256 $FA --freeze-kind external_fixed --stages '<s1>;<s2>,<s3>'
git -C $CODE add configs/global && git -C $CODE commit -m "inbreast: final configs, campaign (a)"   # commit Xa
$G seal-campaign $N --pools-sha256 $P --code $CODE --freeze-sha256 $FA --freeze-kind external_fixed \
   --stages '<s1>;<s2>,<s3>' $Q --image-id $IMAGE --environment-probe $PRIV/probe.json --sealed-at <UTC>
$G anchor $N --pools-sha256 $P --code $CODE $Q --kind campaign --document-sha256 $KA          # C_campaign (a)
```

- As seeds finais são as mesmas nas duas campanhas, e disjuntas das seeds de partição e de
  HPO.
- Só o estágio 1 (`<s1>`, 50 jobs) é autorizado (Q13).
- Anote `Xa = git -C $CODE rev-parse HEAD` e o `commit` impresso pela âncora da campanha
  (`C_campaign (a)`). O recibo privado fica em `anchor-campaign-$KA.json`.
- Os commits de configs (Xa, e depois Xh e Xb) precisam existir no checkout de cada worker,
  porque o worker roda no commit do job. Leve-os ao remoto de código (eles só contêm hashes
  e nomes de arquivos privados, como os configs já versionados) ou ao clone local do worker,
  sem reescrever histórico.

### C2. Campanha (b): HPO exploratório em D (4 lr, 1 seed)

`$PRIV/search-space.json`:
`{"grid":[{"path":"optimizer.parameters.lr","values":[3e-05,0.0001,0.0003,0.001]}],
"default":{"optimizer.parameters.lr":0.0001},"seeds":[<seed HPO>],"budget":{"max_candidates":6,
"max_seeds":2,"max_epochs":20,"job_timeout_s":<s>,"gpu_minutes_cap":<min>},"failure_policy":"require_all_defined"}`.
O `gpu_minutes_cap` faz parte do teto de 8 GPU-h.

```bash
$G hpo-view $N --pools-sha256 $P                                                  # -> V (hpo_view_sha256)
$G d-view   $N --pools-sha256 $P --view-sha256 $V --views-root $VIEWS              # só imagens de D
$G search-space $N --pools-sha256 $P $RT --spec $PRIV/search-space.json --color grayscale_to_rgb   # -> S
$G build-plan $N --pools-sha256 $P $RT --search-space-sha256 $S --view-sha256 $V --final-seeds <s1,s2,s3>   # -> PL, commit P
$G materialize-hpo $N --pools-sha256 $P --code $CODE --search-space-sha256 $S --plan-sha256 $PL --view-sha256 $V
git -C $CODE add configs/global && git -C $CODE commit -m "inbreast: HPO configs"   # commit Xh (só configs/global/**)
$G seed-hpo $N --pools-sha256 $P --code $CODE --search-space-sha256 $S --plan-sha256 $PL --view-sha256 $V $Q \
   --image-id $IMAGE --environment-probe $PRIV/probe.json
```

O `seed-hpo` imprime `code_commit` (= Xh), `plan_commit` (= P) e `code_identity_sha256`.
Ele recusa nestes casos:
- `src/`, `scripts/`, `docker/` ou `pyproject.toml` mudaram entre P e Xh (Q6/B1);
- a imagem ou o probe não reproduzem o `environment_sha256` do plano;
- um config commitado difere do plano;
- a campanha (a) não está ancorada;
- já existem jobs HPO destes pools em outro commit.

Workers HPO: checkout em `Xh`, `settings` privados com `"phase": "hpo"`, `"views": "$VIEWS"`,
`"data": "$INB"` e `"datasets": ["inbreast"]`, mais os campos de sempre (`configs/worker.example.json`).
O worker HPO confere, antes do claim e de novo antes de criar o contêiner, que a imagem e o probe
que ele mesmo executou reproduzem o `environment_sha256` do plano. Outra imagem, ou outro probe
(por exemplo, outra capacidade de GPU), faz o worker parar com `code_identity_mismatch`. Nesse caso, **pare**:
não troque o plano nem o probe.

```bash
./scripts/worker.sh --settings $PRIV/worker-hpo.json --phase hpo        # um treino por vez; tmux
```

**Retries de HPO: regra fixa do operador, enquanto a política não for decidida.**
- Todo job HPO que terminar `failed`/`cancelled`/`interrupted` **sem** `predictions.json`
  recebe **exatamente um** retry manual (procedimento em **C2.1**; retry **não** é `--resume`),
  **antes** do primeiro `collect-hpo`.
  Isso vale para todos, sem exceção, e sem abrir os `results.json` dos outros jobs.
- Se o retry falhar de novo, a linha fica `failed`.
- Um job não aprovado que **já escreveu** predições faz o `collect-hpo` recusar com
  `collection_invalid` (B-1''). **Pare e escale.**
- A primeira coleta completa de um plano é vinculante: ela grava o pin
  `hpo-evidence-pin-<plan>.json`, e qualquer coleta posterior com outra evidência é
  recusada com `collection_invalid`. **Pare e escale**; não há nova rodada.
- O código ainda não impõe a regra do "exatamente um retry". Essa é uma decisão pendente
  (`cloud-context.md` §5).

Quando os 4 jobs estiverem terminais (o `collect-hpo` recusa com `collection_incomplete`
enquanto algum faltar):

```bash
C="--code $CODE --search-space-sha256 $S --plan-sha256 $PL --view-sha256 $V $Q --image-id $IMAGE \
   --environment-probe $PRIV/probe.json --runs $RUNS --views-root $VIEWS --configs-commit <Xh> \
   --settings $PRIV/worker-hpo.json"
$G collect-hpo $N --pools-sha256 $P $C                                                   # -> EV
$G seal-freeze $N --pools-sha256 $P $C --evidence-sha256 $EV --frozen-at <UTC>          # -> FB
$G anchor $N --pools-sha256 $P $C --kind recipe_freeze --freeze-kind hpo_d --document-sha256 $FB   # C_freeze (b)
```

- O vencedor sai do ranking pré-declarado: maior média, e no empate fica o `c00`, o default.
  Nenhuma opção escolhe o vencedor.
- O `seal-freeze` e a âncora do freeze `hpo_d` recalculam a evidência a partir da fila
  **atual** e exigem que ela seja igual ao pin da primeira coleta completa e ao `EV` do
  freeze. Se a fila mudou depois do pin, os dois recusam com `collection_invalid`.
- Linhas `missing`, `failed` ou `timeout` são explícitas, nunca zero.
- Se nenhum candidato ficar definido, o `seal-freeze` falha. **Pare**: não há nova rodada.

#### C2.1. Retry manual de um job HPO (não é `--resume`)

São duas operações diferentes. Não use uma no lugar da outra.

| | Retry manual | `--resume` |
|---|---|---|
| O que faz | Devolve o job terminal a `pending`. O próximo claim cria uma tentativa **nova** (novo `attempt_id`, novo token, novo diretório) e o treino recomeça **do zero**. | Continua a **mesma** tentativa (mesmo `attempt_id`, token, diretório e checkpoint) a partir do checkpoint. |
| Estados | `failed`, `cancelled` ou `interrupted`. Nunca `succeeded`, `claimed` ou `running`. | Só `interrupted`, com checkpoint válido. |
| Como | `GitQueue.retry(job_id, expected_version, reason)` (`docs/queue.md`, "Manual recovery and restart"). **Não há comando de CLI para o retry.** | `./scripts/worker.sh ... --resume` (`README.md`). |
| Registro | A fila grava `manual_retry` com o motivo e o `previous_attempt_id`. | A fila grava um evento `running` com `manual_resume` na mesma tentativa. |

O retry da regra provisória acima é só o retry manual. Se um job HPO `interrupted` com
checkpoint válido deve ser retomado com `--resume` em vez de refeito, a decisão R
(`cloud-context.md` §5) ainda não diz se isso conta como o retry permitido: **registre qual
operação foi usada e pergunte ao Eduardo.**

Procedimento do retry manual (um job por vez):

**Passo 1 — Confira o estado e as predições.** O job tem de estar `failed`, `cancelled` ou
`interrupted`. Em `$RUNS/<job_id>/<attempt_id>/` não pode existir `predictions.json` nem
`results.json`. Se existir, **pare e escale**: o `collect-hpo` recusa esse job com
`collection_invalid` (B-1''), e também recusa um job repetido cuja tentativa anterior já tinha
`predictions.json`. Não apague nada.

**Passo 2 — Confirme que a execução parou.** O worker daquele job foi encerrado, e o contêiner
`atlas-<attempt_id>` (o nome é o `attempt_id` de 32 hex) está parado ou ausente:
`docker inspect atlas-<attempt_id> --format '{{.State.Status}}'`. Um contêiner `running`
significa parar aqui. A fila nunca reatribui trabalho por tempo ou heartbeat.

**Passo 3 — Preserve as evidências antes do retry.** Copie para um diretório privado de
preservação o diretório da tentativa (`docker.log`, `runtime.json`, `completed.json`,
`failed.json`, checkpoint, telemetria), o journal `$RUNS/worker-<GPU uuid>/active.json` com o
`events-pending.json` da tentativa e o `docker logs` do contêiner. **Não apague o journal, o
clone da fila, o contêiner nem o diretório da tentativa.** O `collect-hpo` procura
`predictions.json` nas tentativas anteriores desse diretório.

**Passo 4 — Leia a versão atual do job.** O retry exige a `version` exata do registro, e o
`cli status` mostra só o estado:

```bash
python3 - "$ADMIN" "$REMOTE" <<'EOF'
import sys
from augment_atlas.queue import GitQueue
for job in GitQueue(sys.argv[1], sys.argv[2]).list_jobs():
    if job["spec"]["phase"] == "hpo":
        print(job["job_id"], job["state"], job["attempt_id"], job["version"])
EOF
```

**Passo 5 — Aplique o retry** com o `job_id` e a `version` do passo 4 e um motivo curto. O
motivo vai para o histórico da fila (um branch Git): sem caminhos, hostnames, credenciais nem
resultados. Um motivo vazio é recusado.

```bash
python3 - "$ADMIN" "$REMOTE" <job_id> <version> "infrastructure failure before predictions" <<'EOF'
import sys
from augment_atlas.queue import GitQueue
job = GitQueue(sys.argv[1], sys.argv[2]).retry(sys.argv[3], int(sys.argv[4]), sys.argv[5])
print(job["job_id"], job["state"], job["version"])
EOF
```

`QueueConflict` ("state version changed") significa que o registro mudou: releia o passo 4 e
reavalie. `InvalidTransition` significa que o estado não permite retry: pare.

**Passo 6 — Confirme.** O job volta `pending`, sem `attempt_id`, e o histórico traz o evento
`manual_retry` com o `previous_attempt_id`. Anote no recibo privado: `job_id`, tentativa
anterior, motivo e versão.

**Passo 7 — Religue o worker sem `--resume`.** Se a tentativa anterior terminou `interrupted`,
o `active.json` dela continua em `$RUNS/worker-<GPU uuid>/` e o worker se recusa a seguir sem
`--resume` (`Attempt interrupted; use explicit --resume`). Depois dos passos 2 e 3, **mova**
(não apague) esse `active.json` para o diretório de preservação. Numa tentativa `failed` que o
worker encerrou, o journal já foi removido; qualquer `active.json` que sobrar (por exemplo, de
um `cancelled` publicado por fora) recebe o mesmo tratamento.

O retry não cria exceção para predições já escritas, não escolhe o job a repetir pelo
resultado de outros jobs e não substitui a decisão R.

### C3. Configs finais de (b) e campanha (b)

O diretório `configs/global/inbreast/final/` comporta uma materialização por commit.

```bash
git -C $CODE rm -r configs/global/inbreast/final && git -C $CODE commit -m "inbreast: drop final configs (a) from the tree"
$G materialize-final $N --pools-sha256 $P --code $CODE --freeze-sha256 $FB --freeze-kind hpo_d --stages '<s1>;<s2>,<s3>'
git -C $CODE add configs/global && git -C $CODE commit -m "inbreast: final configs, campaign (b)"   # commit Xb
$G seal-campaign $N --pools-sha256 $P --code $CODE --freeze-sha256 $FB --freeze-kind hpo_d --stages '<s1>;<s2>,<s3>' \
   $Q --image-id $IMAGE --environment-probe $PRIV/probe.json --sealed-at <UTC> --search-space-sha256 $S
$G anchor $N --pools-sha256 $P --code $CODE $Q --kind campaign --document-sha256 $KB            # C_campaign (b)
```

### C4. Jobs finais (só depois das duas âncoras de campanha)

Cada campanha é semeada e executada no **seu** commit:

```bash
git -C $CODE checkout <Xa> && $G seed-final $N --pools-sha256 $P --code $CODE $Q --campaign-sha256 $KA
git -C $CODE checkout <Xb> && $G seed-final $N --pools-sha256 $P --code $CODE $Q --campaign-sha256 $KB
```

Workers finais: o checkout do worker fica no commit da campanha. Os `settings` levam
`"phase": "final"`, `"data": "$INB"`, `"campaign_sha256"` = KA (ou KB) e `"campaign_commit"`
= o `commit` da âncora da campanha correspondente, além de `queue_checkout` e `remote`.
O worker refaz a auditoria da campanha e da cronologia antes de cada claim.

```bash
./scripts/worker.sh --settings $PRIV/worker-final-a.json --phase final
```

Interrupção (guardião, SIGTERM, temperatura):
- o worker pede checkpoint no fim da época;
- depois de confirmar que o contêiner parou, retome com os mesmos settings e `--resume`
  (`README.md`). Isso continua a mesma tentativa; **não** é o retry manual (C2.1), que cria uma
  tentativa nova;
- nunca apague o journal, o clone da fila ou o contêiner antes da reconciliação.

### C5. Cronologia, relatórios e exportação

```bash
$G audit-chronology $N --pools-sha256 $P --code $CODE $Q --campaign-sha256 $KA     # checkout em Xa
$G report $N --pools-sha256 $P --code $CODE $Q --campaign-sha256 $KA --runs $RUNS --settings $PRIV/worker-final-a.json
# repita com KB no checkout Xb
$G summary $N --pools-sha256 $P --report-sha256 <R>                                # público, sem IDs
$G export-article $N --pools-sha256 $P --report-sha256 <R> --out <diretório fora do repositório>
$G compare $N --pools-sha256 $P --report-a <Ra> --report-b <Rb> --out <diretório fora do repositório>
```

O relatório é sempre `descriptive_only`, `exploratory`, com as flags em `false`.
- Células sem artefato aparecem como `missing` (`artifact_not_collected`).
- Basta um artefato contaminado para o relatório inteiro falhar.

## 4. Condições de parada (todas as partes)

- Qualquer exit 2 da CLI global: parar e ler o código. Os códigos estão em
  `global-next-integration.md` §6.11 e em `global-cli.md`.
- `identifier_leak`, `split_audit_failed`, `code_identity_mismatch`, `collection_invalid`,
  `chronology_invalid`, `final_job_before_seal`, `hpo_job_after_freeze`,
  `external_recipe_contaminated`, `metric_recompute_mismatch`: **parar e escalar**.
  Nunca contornar.
- Teto de 8 GPU-h do piloto, guardião V2 acionado, `max_temperature_c`, reserva de disco
  (20 GiB): parar os workers e reportar o estado real.
- `audit-chronology` com `warnings` de inversão de horário: registrar. É evidência de apoio
  (relógios), não aprovação.

## 5. Hashes e commits a registrar (privados; no Brain só os hashes)

| Item | Onde aparece |
|---|---|
| `COMMIT`, `IMAGE`, sha256 dos pesos | §1 |
| identidade do config ISIC (`605233a8…`), sha256 da emenda | `make-pools` (`acquisition_config_sha256`, `split_amendment_sha256`) |
| `global_manifest_sha256`, `pools_sha256` (ISIC e INbreast) | `make-pools` / `import-pools` |
| `records_sha256` do INbreast (387) | `identity` |
| FA, KA, Xa, C_freeze(a), C_campaign(a) | C1 |
| V, S, PL, P, Xh, `code_identity_sha256`, EV, FB, C_freeze(b) | C2 |
| Xb, KB, C_campaign(b) | C3 |
| R(a), R(b), comparação | C5 |

## 6. Estados esperados

| Depois de | Estado |
|---|---|
| B1 | `READY.json` presente; divergência dos 8 pares `accepted: true` |
| B2 | pools ISIC selados e ancorados; nenhum job ISIC |
| C1 | 1 âncora `recipe_freeze` (external_fixed), 1 âncora `campaign`; nenhum job |
| C2 | 4 jobs HPO terminais; evidência coletada; âncora `recipe_freeze` (hpo_d) |
| C3 | 2 campanhas seladas e ancoradas; nenhum job final |
| C4 | 50 + 50 jobs finais (estágio 1); executados até terminar ou até o teto de 8 GPU-h |
| C5 | 2 relatórios privados, resumo público, exportação e comparação descritiva |

## 7. O que este piloto não permite afirmar

Não permite afirmar:
- resultado confirmatório;
- generalização por paciente: os grupos vêm do nome do arquivo e o PatientID não foi
  comprovado;
- teste intocado: o piloto anterior já examinou a base;
- que uma augmentation melhora a mamografia;
- otimalidade da receita;
- independência entre folds ou seeds;
- comparação com a literatura;
- utilidade clínica.

O artigo **não** está completo nem pronto para quatro datasets.
