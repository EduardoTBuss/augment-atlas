# Contexto do trabalho na nuvem — estado em 23/10/2026 (correção dos achados da revisão dos PRs #3 e #4)

Este arquivo existe para que qualquer IA, local ou na nuvem, saiba o que já foi feito, o que
está pendente e qual é o próximo passo sem reler toda a conversa. Leia junto com:
- [`cloud-handoff.md`](cloud-handoff.md): regras duras;
- [`x299-runbook.md`](x299-runbook.md): procedimento do servidor;
- [`global-cli.md`](global-cli.md): referência dos comandos.

O repositório é público: aqui só há código, decisões e contagens de testes. Nenhum dado
privado, caminho, host, credencial ou resultado de campanha.

Atualize este arquivo ao fim de cada sessão de trabalho, mudando a data do título.

**Distinção que vale para tudo abaixo:** "verificado" aqui quer dizer *verificado na nuvem
com fixtures sintéticas*. Nada foi executado no X299, com GPU ou com dados reais. A
execução real é do Eduardo, pelo runbook.

---

## 1. Resumo em uma tela

**Escopo desta etapa:** INbreast e ISIC 2020. mBRSET e CheXpert aguardam o acesso
PhysioNet. Ficam fora o caminho ordinal (mBRSET) e o multilabel (CheXpert).

**Código pronto e testado só com dados sintéticos:**
- as duas campanhas do piloto INbreast, (a) receita externa fixa e (b) HPO exploratório,
  na ordem decidida;
- os pools D/E do ISIC a partir da verificação offline já existente;
- tudo via `python -m augment_atlas.global_cli`, e workers com
  `./scripts/worker.sh --phase smoke|hpo|final`.

**Decisões de 22/10/2026 aplicadas:**
- **B1:** identidade dos jobs HPO, opção (i);
- **Q6:** confirmada;
- **S1/S2:** seeds e regras do split do ISIC, numa emenda pré-registrada;
- **os 8 pares extras de duplicatas do ISIC.**

**Ainda não pode rodar:**
- **HPO e treino final do ISIC:** falta o N de pacientes da subamostragem (Q9) e a
  calibração do orçamento de GPU;
- **o piloto INbreast:** no servidor, exige antes o Gate V4 e o endurance de 5h30
  aprovados no Brain.

**Invariantes:**
- **Nada foi treinado nem executado em servidor.** `scientific_status` continua
  `"provisional"`, e `execution_verified` e `scientific_eligible` continuam `false`.
- **Nada foi mergeado em `main`.** Os dois PRs estão empilhados e em draft (§2).
- O artigo **não** está completo nem pronto para quatro datasets.

## 2. Mapa de branches e PRs

| PR | Branch | Base | Conteúdo | Estado |
|---|---|---|---|---|
| EduardoTBuss/augment-atlas#3 | `claude/tarefa-c-merge-branches-pwmlw1` | `main` | Integração (tarefa C): as 5 branches de feature numa árvore só | draft, sem conflito |
| EduardoTBuss/augment-atlas#4 | `claude/magical-bardeen-vkyubk` | branch do #3 | Sentinel F2, CLI global, coletor final, gerador de pools, B1/HPO, emendas ISIC, runbook, este arquivo | draft, sem conflito |
| EduardoTBuss/augment-atlas#1, #2 | `wip/s4-telemetry-20261021`, `feature/global-protocol-20261016` | `main` | Já contidos no #3 | abertos |

Ordem de merge: **#3, depois #4**. Depois do merge do #3, os PRs #1 e #2 podem ser
fechados, porque o conteúdo deles já está no #3.

Branches de origem integradas no #3 (merge `--no-ff`, nesta ordem):

| Branch | Commit |
|---|---|
| `feature/global-protocol-20261016` (S0–S3 + S2) | `12b4ee3` |
| `wip/s4-telemetry-20261021` (S4 + telemetria de treino) | `2960a5e` |
| `fix/cpu-isa-avx2-20261020` (limite AVX2) | `d812ba3` |
| `feature/fleet-20261019` | `cff5d81` |
| `feature/isic-acquisition-20261019` | `e956b5e` |

## 3. Decisões do Eduardo (22/10/2026) e como foram aplicadas

| Decisão | Aplicação |
|---|---|
| **B1**, opção (i): configs HPO commitados depois do plano se só `configs/global/**` mudar; o commit continua nos hashes | `global_hpo.collect_hpo_evidence(..., code_identity=, code_repo=)` exige a evidência Q6 de `plan.runtime.commit` (P) até o commit dos configs (X) e refaz as identidades dos jobs com X. Sem evidência, nada muda, e jobs em outro commit falham fechados (`search_space_mismatch`). Cada attempt aprovado precisa registrar a imagem pinada. `global_cli seed-hpo`/`collect-hpo` exigem e re-derivam a evidência do Git |
| **Q6** confirmada (`same_code_identity_v1`) | Regra aplicada aos finais (já existia) e agora também ao HPO. **O token `Q6_POLICY_STATUS` continua `"pending_eduardo_confirmation"`**, porque três testes legados fixam essa string (`test_global_campaign.py:107`, `:326`, `test_global_campaign_runtime.py:127`) e testes antigos não são editados. Trocar o token muda o hash das campanhas e exige decisão explícita de atualizar esses testes. A confirmação está registrada aqui e em `global-cli.md` |
| **S1**: seeds explícitas, evaluation distinta, D=0,20, f_sel=0,30, sem novas seeds | `configs/acquisition/isic2020-v1-splits-amendment-20261022.json`: development = selection = 20261019 (a `split_seed` pré-registrada antes do download), evaluation = 20261022, limiares 15/50, `before_any_score: true`. `make-pools` recusa qualquer divergência da emenda ou do config |
| **S2**: quartis nearest-rank, D_select nos estratos de D, meio-para-cima exato, contador único de folds | Implementado em `global_splits.py` e escrito na emenda (`rules`) |
| **S3** (mBRSET) adiada | Nada implementado; `global_splits` recusa tarefas não binárias |
| **8 pares extras byte-idênticos** | Entrada em `accepted_divergences` de `configs/acquisition/isic2020-v1.json` (`duplicates.identical_originals_not_listed`, `observed: 8`, `decided_by: Eduardo`). É uma chave operacional e a identidade do config não muda. O `verify` já une pares de SHA idêntico no union-find e aplica `keep-min-image-name-v1`. Não há novo download |
| **INbreast**: duas campanhas exploratórias; selar antes do 1º final; a campanha externa antes de qualquer HPO | `seed-hpo` recusa, num dataset exposto (`prior_exposure ≠ none_documented`), enquanto a âncora da campanha `external_fixed` não existir. A cronologia recusa finais antes das âncoras. A ordem completa está testada de ponta a ponta |
| Já decididos, não perguntar de novo | Licença CC BY-NC e atribuição do ISIC, tarefa binária, AUROC primária, derivado 512 px, agrupamento padrão, frações, autorização de download |

**Por que a emenda do ISIC é um arquivo separado.** O bloco `splits` de `isic2020-v1.json`
faz parte da identidade gravada no ledger do download (`605233a8…`). Editar esse bloco faria
o `verify` recusar os dados já baixados. Além disso, `acquisition/splits_global.py`, o split
de seed única do `verify`, é fixado por testes antigos. A saída `splits/` do `verify` passa a
ser um diagnóstico pré-emenda, e quem gera os pools é `make-pools`.

## 4. O que foi feito (cronológico)

### 4.1 Integração (PR #3)

- **Conflitos resolvidos sem mudar comportamento:**
  - `runtime.py`: a checagem de ISA continua primeiro, e a montagem da D-view foi mantida;
  - `cli.py`: `recipe_identity` vem de `contracts`;
  - `queue.py`: âncoras S3 e documentos do fleet;
  - `docs/global-next-integration.md`: blocos S2 e S4.
- **Revisões independentes** confirmaram as resoluções.

### 4.2 PR #4

**1ª sessão de 22/10:**

| Commit | O quê |
|---|---|
| `c4aefd2` | **Sentinel F2:** ferramenta ausente do PATH sai como `tool_absent` mesmo quando o exec responde EACCES |
| `4f8831e` | **`global_cli.py` e `global_collect.py`:** CLI e coletor de jobs finais |
| `8425f6f` | **Correções da 1ª revisão:** `results.json` checksummado, modo de validação explícito, varredura de IDs |
| `e6aa7f3` | **`global_splits.py` e `make-pools`:** gerador `global-dev-eval-v1` |
| `7510b38` | **Correções da 2ª revisão:** frações e limiares presos ao config, meio-para-cima exato |
| `7a96d83` | Primeira versão deste arquivo |

**2ª sessão de 22/10 (esta):**

| Commit | O quê |
|---|---|
| `324d4b9` | **B1 no coletor HPO** (evidência Q6 de P até o commit dos configs); gancho `--phase hpo\|final` do worker em `cli.py`; `tests/test_global_hpo_b1.py` |
| `f166799` | **Caminho HPO na CLI:** `search-space`, `build-plan`, `materialize-hpo`, `seed-hpo`, `collect-hpo`, `seal-freeze`. `tests/test_global_cli_hpo.py` cobre a campanha (b) de ponta a ponta e a ordem completa do INbreast com as duas campanhas, a comparação incluída |
| `9ae9c02` | **ISIC:** emenda de split S1/S2, divergência aceita dos 8 pares, `split-spec`, `make-pools` exigindo a emenda; `tests/test_isic_amendment.py` (e2e: aquisição sintética → verify → pools selados) |
| `b5ee1c9` | **`docs/x299-runbook.md`** e a atualização de `global-cli.md` e `README.md` |
| `0fe04b6` | Primeira versão deste arquivo para a 2ª sessão |
| `4f3fd09` | **Correções da auditoria integrada** (detalhes logo abaixo) |
| `bcf27ca` | Registro da auditoria neste arquivo |
| `5bbae41` | **Correções da re-revisão** (detalhes logo abaixo) |
| `47402d5` | Registro da re-revisão neste arquivo |
| `9d8125c` | **Correções da 3ª revisão** (detalhes logo abaixo) |
| (este) | Esta versão deste arquivo |

**Auditoria integrada independente** (contexto novo, somente leitura), sobre os commits
`324d4b9`..`b5ee1c9` e a integração.

**Bloqueador corrigido em `4f3fd09`:**
- **B-1:** o operador podia escolher o vencedor do freeze `hpo_d` escolhendo um snapshot da
  evidência. Agora:
  - o `collect-hpo` recusa com `collection_incomplete` enquanto algum job HPO faltar ou não
    for terminal;
  - o `seal-freeze` recalcula a evidência a partir da fila **atual** e só aceita a igual a
    esse recálculo.

**Maiores corrigidos:**
- **M-1:** a imagem não era pinada. Agora o `environment_sha256(imagem, probe)` precisa
  igualar o do plano ou do freeze no `seed-hpo`, `collect-hpo`, `seal-freeze` e
  `seal-campaign`.
- **M-2:** a emenda do ISIC não era pinada. Agora o `make-pools` só lê o config e a emenda
  como arquivos rastreados e sem alteração de um `--code` limpo, presos ao `READY.json`, e
  exige que a spec seja igual à derivada da raiz verificada.

**Menores corrigidos:**
- o `seed-final` num dataset exposto exige as duas âncoras de campanha;
- o `seed-hpo` recusa jobs HPO em outro commit;
- testes do gancho `--phase` do worker;
- `DATA` no runbook, e todas as flags do runbook conferidas contra o parser;
- documentos desatualizados.

**Registrados sem correção (§6):**
- o freeze não grava o commit X dos configs HPO; só o manifesto privado de coleta grava;
- a decisão 7 é aplicada pela CLI e pela cronologia do lado da campanha (b), não por um
  gancho do worker.

**Re-revisão independente de `4f3fd09`.** M-1 e os menores estavam corrigidos, mas ela achou:
- **B-1' (bloqueador):** um freeze selado antes de um retry ainda podia ser ancorado. O
  operador também podia decidir fazer o retry depois de ver a evidência completa.
- **M-2a:** os registros do `make-pools` não estavam presos ao `manifest.csv` do READY.
- **M-2b:** um commit só local, ou um arquivo marcado `skip-worktree`, passava por emenda
  pré-registrada.
- **Menor:** o coletor final não conferia a imagem dos attempts.

**Corrigidos em `5bbae41`:**
- **A primeira coleta HPO completa de um plano é vinculante.** Ela grava o pin privado
  `hpo-evidence-pin-<plan>`, e qualquer coleta posterior com outra evidência recebe
  `collection_invalid`. Os retries de infraestrutura ficam antes da primeira coleta.
- **`seal-freeze` e a âncora do freeze `hpo_d`** recalculam a evidência e exigem igualdade
  com o pin e com o freeze.
- **`make-pools`:**
  - exige `--source` igual ao `manifest.csv` do READY;
  - lê o config e a emenda com `git show HEAD:<path>`;
  - exige o HEAD numa branch remota.
- **Coletor final:** exige a imagem da campanha em cada attempt.

**Limite declarado:** o pin fica no store privado do operador. Ele protege contra erro e
escolha posterior, não contra quem apaga arquivos privados (spec R9; as flags continuam
falsas).

**3ª revisão independente (de `5bbae41`).** Ela confirmou fechadas todas as sequências
anteriores:
- freeze antigo ancorado depois de um retry;
- resume depois da coleta;
- manifesto alternativo;
- `skip-worktree`;
- commit só local;
- imagem dos finais.

Achou, e foi corrigido em `9d8125c`:
- **B-1'' (bloqueador):** um attempt HPO que gravou o score e depois foi encerrado como
  `failed`/`interrupted` virava uma linha de falha comum. Agora é `collection_invalid`.
- **M-2b (residual):** "branch remota" era fraco demais. Agora o config e a emenda precisam
  ser os mesmos blobs de `origin/main`.
- **Checksum do pin** passa a ser verificado.
- **Documentos** que prometiam mais do que o código faz foram corrigidos.

**Ficou aberto (decisão do Eduardo, §5.1):** a política de retry de HPO. Antes da primeira
coleta, o operador ainda decide se refaz uma falha de infraestrutura enquanto os scores dos
outros jobs já estão em disco. O runbook fixa uma regra provisória: exatamente um retry para
toda falha sem predições. O código ainda não impõe essa regra.

**`9d8125c` não passou por nova revisão independente.** Cada correção tem teste próprio, que
falha sem ela.

### 4.3 Sessão de 23/10/2026: achados da revisão dos PRs #3 e #4

**Identificação:** `TASK_ID: augment-atlas-review-pr3-pr4-2026-10-23`.

**Heads confirmados no GitHub antes de editar** (os dois PRs seguem em draft, na ordem #3 → #4,
o #3 é a base do #4):

| PR | Branch | Head |
|---|---|---|
| #3 | `claude/tarefa-c-merge-branches-pwmlw1` | `34627c4d2ce71ba236d4e94338320670eb44d34d` |
| #4 | `claude/magical-bardeen-vkyubk` | `c537b3c68bbde469f615eee5f56a15f15557e16f` |

**Lacuna de rastreabilidade (a preencher pelo Eduardo).** O anexo do handoff de revisão não
chegou a esta sessão: não está na conversa, no repositório, nos comentários nem nas revisões dos
PRs. O contrato foi o texto da tarefa. Por isso **os SHAs que o revisor de origem revisou e o
nome desse revisor não estão registrados aqui**; só os heads atuais acima, confirmados. Quem
tiver o handoff deve acrescentar essas duas linhas. Nada foi inferido.

**Onde estão os commits.** Em `claude/nifty-feynman-x7g9hi`, criada a partir do head do #4
(`c537b3c`). **Os branches dos PRs não foram alterados** e nada foi enviado para eles: a sessão
só tinha autorização para empurrar esta branch. Para entrar no PR #4 basta mergear (ou
avançar) esta branch nele. O commit do F2 mexe só em arquivos que já existem no #3 e foi
aplicado em um worktree descartável sobre `34627c4` por `cherry-pick`, sem conflito, com os
testes do runtime e do HPO passando (174 testes); ele pode ser levado ao #3 e depois ao #4.

| Achado | Estado | Onde |
|---|---|---|
| **F1** guard B-1'' depois dos atalhos de timeout/cap | **corrigido** | `global_hpo.collect_hpo_evidence`, commit `c2ff5b3` |
| **F2** imagem e probe do worker HPO contra `plan.runtime.environment_sha256` | **corrigido** | `global_hpo.audit_worker_environment`, gancho em `runtime.py`, commit `4b590cb` |
| **F3** runbook confundia retry com `--resume` | **corrigido** | `x299-runbook.md` C2.1, `README.md`, commit `22fe609` |
| **V1** `test_two_workers_with_tiny_spool_apply_backpressure` | **não reproduzido; nada alterado** | §4.3.2 |
| **R1** política de retry de HPO | **pendente, decisão do Eduardo; nada implementado** | §6.1, linha R |

#### 4.3.1 O que cada correção faz

- **F1.** A checagem "attempt não aprovado que já escreveu `predictions.json` ou `results.json`"
  agora roda **antes** dos atalhos de timeout por job e de teto de GPU-minutos. Antes, um attempt
  `failed`/`cancelled`/`interrupted` que passasse do timeout ou do teto virava uma linha
  `timeout` comum e o candidato era descartado depois de o próprio score ter aparecido. Agora é
  `collection_invalid`, antes de qualquer pin ou freeze (o pin e o freeze só existem depois de
  uma coleta bem-sucedida). A detecção usa `lexists`, então um `results.json` que seja symlink
  quebrado também conta. Preservado: o timeout legítimo de attempts `succeeded` (a linha
  `timeout` continua sem score), as falhas sem scores (linha `failed`) e o teto monótono.
- **F2.** `DockerRuntime.check_environment(spec)` (a lógica está em
  `global_hpo.audit_worker_environment`) compara `environment_sha256(imagem do worker, probe que
  ele mesmo executou)` com `plan.runtime.environment_sha256`. O plano é o documento privado do
  data root e só vale se o hash canônico dele for o `plan_sha256` do job. O `runtime.worker`
  chama a checagem depois de `config_for` e **antes de `queue.claim`**, e de novo **antes de
  `runtime.create`** no caminho de recuperação. Imagem ou probe divergente, ou worker sem probe
  registrado, é `code_identity_mismatch`; plano ausente ou adulterado é `plan_invalid`. Jobs
  `final` e `smoke` não são afetados (o final já era preso à imagem da campanha). `PROBE`, a
  fórmula de `environment_sha256` e os hashes de Q6 não mudaram.
  - **Por que não em `config_for`/`create`:** os testes legados do runtime montam o
    `DockerRuntime` com imagem e probe de mentira, que não reproduzem o hash do plano. Em um
    experimento descartado, a checagem dentro de `global_config_for` quebrou 5 testes legados
    (`test_global_runtime`: `test_hpo_config_for_checks_everything_on_the_host_without_torch`,
    `test_mock_create_mounts_only_the_d_view_as_data_and_records_it`,
    `test_d_view_listing_is_recomputed_before_mount`,
    `test_hpo_without_d_view_mount_is_isolation_unproven_and_never_falls_back`,
    `test_invalid_hpo_job_fails_real_worker_selection_before_mock_claim`). Como testes antigos não
    são editados, o gancho fica no `worker()`, que é o caminho de produção.
  - **Consequência:** quem chamar `DockerRuntime.create` direto, fora do `worker()`, não passa por
    essa checagem. O `scripts/worker.sh` e a CLI sempre passam pelo `worker()`.
  - **Efeito prático:** o probe inclui a capacidade da GPU. Um worker com GPU de outra
    capacidade que a do probe do plano é recusado, de propósito.
- **F3.** `x299-runbook.md` ganhou C2.1: uma tabela retry manual × `--resume` e um procedimento
  em 7 passos (estado, ausência de scores, confirmação de parada, preservação do journal e dos
  artefatos, leitura da `version` exata, motivo de auditoria, confirmação, e o caso do
  `active.json` de uma tentativa `interrupted`). O runbook apontava o retry para o `README.md`,
  que só descreve `--resume`; não existe comando de CLI para o retry, só
  `GitQueue.retry(job_id, expected_version, reason)`. Os dois trechos Python do runbook foram
  executados contra uma fila Git local sintética (`failed` v3 → `pending` v4, evento
  `manual_retry` com `previous_attempt_id`; versão velha → `QueueConflict`). A regra provisória
  ("exatamente um retry") e a decisão R ficaram como estavam; o runbook diz que ainda não está
  decidido se um `--resume` conta como esse retry.

#### 4.3.2 V1: o teste de backpressure

Teste intacto, sem alterar delays, sem excluir e sem editar nenhum teste legado.

| Condição | Execuções | Falhas |
|---|---|---|
| isolado, sequencial | 3 + 40 | 0 |
| 8 processos em paralelo, 5 rodadas, 4 núcleos | 40 | 0 |
| 1 núcleo (`taskset`) com 3 processos de CPU concorrentes | 12 | 0 |
| módulo `test_acquisition_isic_e2e` inteiro | 1 (19 testes) | 0 |
| suíte completa como root (931 testes, antes das mudanças) | 1 | 0 neste teste |
| suíte completa como `nobody` (952 testes, com as mudanças) | 1 | 0 neste teste |

**Não consegui reproduzir a falha**, e por isso não alterei código nem teste. A medição de
sanidade mostra 63–64 reaberturas do stream por execução contra o limiar do teste (`> 3`), uma
folga de mais de 15 vezes; uma falha por "poucas pausas" é improvável por temporização. Não
tenho a saída original da falha (o anexo não chegou). **Para continuar preciso do traceback ou
da mensagem de asserção, do usuário, do Python e do modo de execução (isolado, suíte inteira,
paralelo) em que ela apareceu.** Sem isso, qualquer correção seria um palpite. Se a causa exigir
editar o teste legado, vou parar e trazer evidências e uma proposta ao Eduardo antes.

#### 4.3.3 Arquivos alterados

| Arquivo | Tipo | Achado |
|---|---|---|
| `src/augment_atlas/global_hpo.py` | código | F1, F2 |
| `src/augment_atlas/runtime.py` | gancho (1 método, 2 chamadas no `worker()`) | F2 |
| `tests/test_global_hpo_f1.py` | novo | F1 |
| `tests/test_global_hpo_f2.py` | novo | F2 |
| `docs/x299-runbook.md`, `README.md`, `docs/cloud-context.md` | documentação | F3, F2, este registro |

Nenhum teste legado foi alterado. Os dois arquivos de teste novos falham sem as correções:
`test_global_hpo_f1.py` teve 16 falhas de 11 testes sem o guard reordenado, e
`test_global_hpo_f2.py` teve 3 falhas e 1 erro de 10 testes sem os ganchos do `worker()`.

## 5. Testes

Comando: `PYTHONPATH=src python3 -m unittest discover -s tests`. Python 3.13, sem torch,
no commit `9d8125c`.

| Execução | Rodados | Passaram | Pulados | Falharam |
|---|---|---|---|---|
| root | 931 | 895 | 35 | 1 (F1) |
| usuário sem privilégios (`nobody`), com o PATH do root | 931 | 897 | 34 | 0 |

- **Pulados:** testes de torch/contêiner, mais (só como root) o teste de exec real de
  EACCES. Pulado não é aprovado.
- **F1:** `test_sentinel.CliAndScriptTests.test_cli_run_report_latest` falha como root
  porque o sentinel recusa root de propósito. **Rode a suíte com um usuário sem
  privilégios.**
- **Testes legados:** os 6 arquivos de teste que diferem de `main` e das branches de
  origem são todos novos neste PR. Nenhum teste antigo foi alterado.
- **O que os testes provam:** só caminhos sintéticos (arquivos reais em diretório
  temporário, repositório Git local, fila Git bare, servidor HTTP em loopback). Não provam
  nada sobre o X299, a GPU, a imagem real ou os dados reais.

## 6. Achados registrados, sem correção nesta etapa

- **Arquivo central:** o worker (`runtime.py`, `_global_config_bytes`) lê config e
  MANIFEST da árvore de trabalho, não do blob do HEAD. O conteúdo continua preso por hash,
  então não há falha aberta. A correção sugerida, num PR separado, é
  `git cat-file blob HEAD:<path>`.
- **Q6:** o token `pending_eduardo_confirmation` continua no código (§3).
- **Ponta do HPO no worker:** desde 23/10 (F2, §4.3) o worker HPO confere a imagem e o probe
  contra o `environment_sha256` do plano, antes do claim e antes de criar o contêiner. Ele
  **ainda não** confere sozinho a identidade de código Q6 entre P e o seu HEAD: essa proteção
  continua no `seed-hpo`, que recusa semear, e no `collect-hpo`, que recusa coletar.
- **Rastro do commit HPO:** o freeze `hpo_d` (schema v1 fechado) não grava o commit X dos
  configs HPO nem o hash da evidência Q6. Eles ficam no manifesto privado
  `hpo-collection-<sha>.json` e no `hpo-code-identity-<sha>.json`.
- **Decisão 7 fora da CLI:** a cronologia da campanha (a) exige jobs HPO depois do
  C_freeze(a), não depois do C_campaign(a). Jobs HPO semeados fora da CLI entre essas duas
  âncoras passariam nessa auditoria. A CLI recusa esse caso.
- **INbreast pools:** se o arquivo privado de 16/10 não estiver no schema v1, a conversão
  das mesmas listas (sem novo sorteio) é uma tarefa de código à parte.

### 6.1 Decisões ainda pendentes

| # | Decisão | Opções |
|---|---|---|
| R | Política de retry de HPO | (i) retries obrigatórios até K tentativas, com K gravado no search space; ou (ii) nenhum retry de HPO, e a falha fica como linha `failed`. Até decidir, vale a regra provisória do runbook (exatamente um retry) |
| Q9 | N de pacientes da subamostragem do ISIC | declarar N antes do split do ISIC usado em treino |
| Q6-token | Trocar o token `pending_eduardo_confirmation` | exige autorização explícita para atualizar três testes legados; muda o hash das campanhas novas |

## 7. Bloqueado por acesso ou decisão

| Item | Bloqueio |
|---|---|
| mBRSET/BRSET | acesso PhysioNet; regra ordinal (S3) adiada |
| CheXpert/MIMIC | acesso; treino multilabel (S6) fora do escopo |
| HPO e finais do ISIC | N da subamostragem por paciente (Q9) e calibração do orçamento de GPU |
| Piloto INbreast no servidor | Gate V4 e endurance de 5h30 pendentes no Brain |

## 8. Próximo passo exato para o Eduardo

1. Revisar e mergear o PR #3, depois o PR #4. Fechar os PRs #1 e #2.
2. No X299, a partir do commit mergeado, seguir [`x299-runbook.md`](x299-runbook.md):
   - **Parte A:** suíte com usuário sem privilégios; imagem, pesos, AVX2, sentinel e smoke;
   - **Parte B:** `acq none verify` com a emenda dos 8 pares, depois `split-spec`,
     `make-pools`, `import-pools`, `audit-pools` e `anchor --kind pools`. Anotar os hashes
     (runbook §5) no Brain;
   - **Parte C:** só depois do Gate V4 e do endurance aprovados.
3. Trazer para a nuvem:
   - os hashes da Parte B (nunca IDs);
   - as contagens do `make-pools`;
   - a decisão do N da Q9 para o ISIC;
   - a decisão R (política de retry de HPO), se houver;
   - se os pools do INbreast já estão no schema v1.

## 9. Como a sessão na nuvem trabalha (para manter o padrão)

- **Laço do handoff:** implementar, rodar testes, revisão independente em contexto novo e
  somente leitura, corrigir os achados, repetir.
- **Testes:** testes antigos nunca são editados. Todo caso adversarial tem teste, e
  mutações são reassinadas para o checksum não esconder a falha.
- **Módulos:** lógica nova vai em módulo novo. Os arquivos centrais só recebem ganchos
  pequenos; o único desta sessão é o `--phase` do worker em `cli.py`.
- **Commits e PRs:** sem trailers. Nada vai direto para `main`. Os PRs ficam em draft até o
  Eduardo revisar.
