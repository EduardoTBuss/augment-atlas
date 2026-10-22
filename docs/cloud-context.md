# Contexto do trabalho na nuvem — estado em 22/10/2026 (2ª sessão do dia)

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

**Re-revisão independente das correções:** em andamento quando este arquivo foi gravado; o resultado entra na próxima atualização.

## 5. Testes

Comando: `PYTHONPATH=src python3 -m unittest discover -s tests`. Python 3.13, sem torch,
no commit `4f3fd09`.

| Execução | Rodados | Passaram | Pulados | Falharam |
|---|---|---|---|---|
| root | 926 | 890 | 35 | 1 (F1) |
| usuário sem privilégios (`nobody`), com o PATH do root | 926 | 892 | 34 | 0 |

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
- **Ponta do HPO no worker:** o worker HPO não confere sozinho a identidade Q6 entre P e o
  seu HEAD. A proteção está no `seed-hpo`, que recusa semear, e no `collect-hpo`, que recusa
  coletar. Um gancho no `runtime.py` seria mais cedo, mas mexe em arquivo central.
- **Rastro do commit HPO:** o freeze `hpo_d` (schema v1 fechado) não grava o commit X dos
  configs HPO nem o hash da evidência Q6. Eles ficam no manifesto privado
  `hpo-collection-<sha>.json` e no `hpo-code-identity-<sha>.json`.
- **Decisão 7 fora da CLI:** a cronologia da campanha (a) exige jobs HPO depois do
  C_freeze(a), não depois do C_campaign(a). Jobs HPO semeados fora da CLI entre essas duas
  âncoras passariam nessa auditoria. A CLI recusa esse caso.
- **INbreast pools:** se o arquivo privado de 16/10 não estiver no schema v1, a conversão
  das mesmas listas (sem novo sorteio) é uma tarefa de código à parte.

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
3. Trazer para a nuvem: os hashes da Parte B (nunca IDs), as contagens do `make-pools`, a
   decisão do N da Q9 para o ISIC, e se os pools do INbreast já estão no schema v1.

## 9. Como a sessão na nuvem trabalha (para manter o padrão)

- **Laço do handoff:** implementar, rodar testes, revisão independente em contexto novo e
  somente leitura, corrigir os achados, repetir.
- **Testes:** testes antigos nunca são editados. Todo caso adversarial tem teste, e
  mutações são reassinadas para o checksum não esconder a falha.
- **Módulos:** lógica nova vai em módulo novo. Os arquivos centrais só recebem ganchos
  pequenos; o único desta sessão é o `--phase` do worker em `cli.py`.
- **Commits e PRs:** sem trailers. Nada vai direto para `main`. Os PRs ficam em draft até o
  Eduardo revisar.
